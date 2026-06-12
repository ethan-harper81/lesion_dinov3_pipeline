"""Standalone SIFT crop extraction + PyTorch VGG16 six-way classification.

This script detects SIFT blobs/keypoints in one image, creates nonoverlapping
2d x 2d crops centered on accepted keypoints, resizes crops to 150 x 150 x 3,
and runs a torchvision VGG16 six-class model on GPU when available.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class CropRecord:
    crop_index: int
    keypoint_x: float
    keypoint_y: float
    diameter: float
    response: float
    octave: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect SIFT blobs, crop nonoverlapping 2d x 2d patches around "
            "keypoints, and classify patches with a torchvision VGG16 model."
        )
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help=(
            "Optional PyTorch checkpoint for the six-class VGG16 classifier. "
            "If omitted, the VGG16 backbone is loaded from torchvision and the "
            "six-class head is randomly initialized."
        ),
    )
    parser.add_argument(
        "--weights",
        default="DEFAULT",
        choices=("DEFAULT", "IMAGENET1K_V1", "none"),
        help="Torchvision VGG16 weights to load before replacing the classifier head.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("sift_vgg16_torch_output"),
        type=Path,
        help="Directory for crops, metadata CSV, predictions CSV, and timing JSON.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        help="Comma-separated class labels. Defaults to class_0,...,class_5.",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="Inference batch size.")
    parser.add_argument("--warmup-runs", default=1, type=int, help="Untimed inference warmups.")
    parser.add_argument(
        "--model-output",
        choices=("logits", "probabilities"),
        default="logits",
        help="How to interpret model outputs before writing class probabilities.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--max-crops",
        default=None,
        type=int,
        help="Optional cap on accepted nonoverlapping crops after response sorting.",
    )
    parser.add_argument(
        "--min-diameter",
        default=4.0,
        type=float,
        help="Minimum SIFT keypoint diameter in pixels.",
    )
    parser.add_argument(
        "--max-diameter",
        default=None,
        type=float,
        help="Optional maximum SIFT keypoint diameter in pixels.",
    )
    parser.add_argument(
        "--sift-nfeatures",
        default=0,
        type=int,
        help="OpenCV SIFT nfeatures. 0 keeps all detected keypoints.",
    )
    parser.add_argument(
        "--sift-contrast-threshold",
        default=0.04,
        type=float,
        help="OpenCV SIFT contrastThreshold.",
    )
    parser.add_argument(
        "--sift-edge-threshold",
        default=10.0,
        type=float,
        help="OpenCV SIFT edgeThreshold.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use torch autocast float16 during CUDA inference.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the VGG16 model when available.",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Skip writing crop PNG files; metadata and predictions are still written.",
    )
    return parser.parse_args()


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detect_sift_keypoints(
    rgb_image: np.ndarray,
    nfeatures: int,
    contrast_threshold: float,
    edge_threshold: float,
    min_diameter: float,
    max_diameter: float | None,
) -> list[cv2.KeyPoint]:
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    keypoints = sift.detect(gray, None)
    keypoints = [
        kp
        for kp in keypoints
        if kp.size >= min_diameter and (max_diameter is None or kp.size <= max_diameter)
    ]
    return sorted(keypoints, key=lambda kp: kp.response, reverse=True)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def keypoint_crop_box(kp: cv2.KeyPoint, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    diameter = max(1.0, float(kp.size))
    side = max(1, int(round(diameter * 2.0)))
    center_x, center_y = kp.pt
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    clipped_x0 = max(0, x0)
    clipped_y0 = max(0, y0)
    clipped_x1 = min(image_width, x1)
    clipped_y1 = min(image_height, y1)
    return clipped_x0, clipped_y0, clipped_x1, clipped_y1


def select_nonoverlapping_keypoints(
    keypoints: Iterable[cv2.KeyPoint],
    image_width: int,
    image_height: int,
    max_crops: int | None,
) -> list[tuple[cv2.KeyPoint, tuple[int, int, int, int]]]:
    selected: list[tuple[cv2.KeyPoint, tuple[int, int, int, int]]] = []
    selected_boxes: list[tuple[int, int, int, int]] = []

    for kp in keypoints:
        box = keypoint_crop_box(kp, image_width, image_height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(boxes_overlap(box, existing) for existing in selected_boxes):
            continue
        selected.append((kp, box))
        selected_boxes.append(box)
        if max_crops is not None and len(selected) >= max_crops:
            break

    return selected


def crop_centered_with_padding(rgb_image: np.ndarray, kp: cv2.KeyPoint) -> np.ndarray:
    diameter = max(1.0, float(kp.size))
    side = max(1, int(round(diameter * 2.0)))
    center_x, center_y = kp.pt
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    height, width = rgb_image.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            rgb_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
        return padded[y0:y1, x0:x1]

    return rgb_image[y0:y1, x0:x1]


def build_crop_tensors(
    rgb_image: np.ndarray,
    selected: list[tuple[cv2.KeyPoint, tuple[int, int, int, int]]],
    out_dir: Path,
    save_crops: bool,
) -> tuple[np.ndarray, list[CropRecord]]:
    crop_dir = out_dir / "crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[np.ndarray] = []
    records: list[CropRecord] = []

    for crop_index, (kp, box) in enumerate(selected):
        crop = crop_centered_with_padding(rgb_image, kp)
        resized = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)
        tensors.append(resized.astype(np.float32))

        if save_crops:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_dir / f"crop_{crop_index:05d}.png"), crop_bgr)

        x0, y0, x1, y1 = box
        records.append(
            CropRecord(
                crop_index=crop_index,
                keypoint_x=float(kp.pt[0]),
                keypoint_y=float(kp.pt[1]),
                diameter=float(kp.size),
                response=float(kp.response),
                octave=int(kp.octave),
                crop_x0=x0,
                crop_y0=y0,
                crop_x1=x1,
                crop_y1=y1,
            )
        )

    if not tensors:
        return np.empty((0, 150, 150, 3), dtype=np.float32), records
    return np.stack(tensors, axis=0), records


def write_crop_metadata(path: Path, records: list[CropRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CropRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    return normalized


def resolve_vgg16_weights(weights_arg: str):
    from torchvision.models import VGG16_Weights

    if weights_arg == "none":
        return None
    if weights_arg == "DEFAULT":
        return VGG16_Weights.DEFAULT
    return getattr(VGG16_Weights, weights_arg)


def build_vgg16_model(checkpoint_path: Path | None, weights_arg: str, device):
    import torch
    from torchvision.models import vgg16

    model = vgg16(weights=resolve_vgg16_weights(weights_arg))
    in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_features, 6)

    loaded_checkpoint = checkpoint_path is not None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        warnings.warn(
            "No --checkpoint was provided. The VGG16 backbone comes from torchvision, "
            "but the six-class classifier head is randomly initialized.",
            RuntimeWarning,
            stacklevel=2,
        )
    model.to(device)
    model.eval()
    return model, loaded_checkpoint


def preprocess_crops_for_vgg(crops: np.ndarray):
    import torch

    batch = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()
    batch = batch.float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (batch - mean) / std


def outputs_to_probabilities(outputs, model_output: str):
    import torch

    if model_output == "probabilities":
        return outputs
    return torch.softmax(outputs, dim=-1)


def batched_predict(
    model,
    crops: np.ndarray,
    batch_size: int,
    device,
    model_output: str,
    mixed_precision: bool,
) -> np.ndarray:
    import torch

    outputs = []
    use_cuda_amp = mixed_precision and device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = preprocess_crops_for_vgg(crops[start : start + batch_size]).to(
                device, non_blocking=True
            )
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                logits_or_probs = model(batch)
                probabilities = outputs_to_probabilities(logits_or_probs, model_output)
            outputs.append(probabilities.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 6), dtype=np.float32)


def write_predictions(
    path: Path,
    records: list[CropRecord],
    probabilities: np.ndarray,
    labels: list[str],
) -> None:
    fieldnames = [
        "crop_index",
        "keypoint_x",
        "keypoint_y",
        "diameter",
        "response",
        "predicted_class",
        "predicted_label",
        "predicted_probability",
    ] + [f"prob_{label}" for label in labels]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, probs in zip(records, probabilities):
            pred_idx = int(np.argmax(probs))
            row = {
                "crop_index": record.crop_index,
                "keypoint_x": record.keypoint_x,
                "keypoint_y": record.keypoint_y,
                "diameter": record.diameter,
                "response": record.response,
                "predicted_class": pred_idx,
                "predicted_label": labels[pred_idx],
                "predicted_probability": float(probs[pred_idx]),
            }
            row.update({f"prob_{label}": float(prob) for label, prob in zip(labels, probs)})
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    labels = args.labels.split(",") if args.labels else [f"class_{i}" for i in range(6)]
    if len(labels) != 6:
        raise ValueError("--labels must contain exactly six comma-separated labels.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    import torch

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    args.out_dir.mkdir(parents=True, exist_ok=True)

    single_image_start = time.perf_counter()
    rgb_image = load_rgb_image(args.image)
    height, width = rgb_image.shape[:2]

    keypoints = detect_sift_keypoints(
        rgb_image=rgb_image,
        nfeatures=args.sift_nfeatures,
        contrast_threshold=args.sift_contrast_threshold,
        edge_threshold=args.sift_edge_threshold,
        min_diameter=args.min_diameter,
        max_diameter=args.max_diameter,
    )
    selected = select_nonoverlapping_keypoints(
        keypoints=keypoints,
        image_width=width,
        image_height=height,
        max_crops=args.max_crops,
    )
    crops, records = build_crop_tensors(
        rgb_image=rgb_image,
        selected=selected,
        out_dir=args.out_dir,
        save_crops=not args.no_save_crops,
    )
    write_crop_metadata(args.out_dir / "crops_metadata.csv", records)
    preprocessing_seconds = time.perf_counter() - single_image_start

    model_load_start = time.perf_counter()
    model, loaded_checkpoint = build_vgg16_model(args.checkpoint, args.weights, device)
    if args.compile:
        model = torch.compile(model)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_start

    warmup_start = time.perf_counter()
    for _ in range(args.warmup_runs):
        if len(crops):
            batched_predict(
                model,
                crops[: min(len(crops), args.batch_size)],
                args.batch_size,
                device,
                args.model_output,
                args.mixed_precision,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    inference_start = time.perf_counter()
    probabilities = batched_predict(
        model,
        crops,
        args.batch_size,
        device,
        args.model_output,
        args.mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    postprocess_start = time.perf_counter()

    if probabilities.shape[1] != 6:
        raise ValueError(f"Expected six model outputs, got shape {probabilities.shape}.")

    write_predictions(args.out_dir / "predictions.csv", records, probabilities, labels)
    postprocess_seconds = time.perf_counter() - postprocess_start
    single_image_seconds_excluding_model_load = (
        preprocessing_seconds + inference_seconds + postprocess_seconds
    )
    end_to_end_seconds_including_model_load = (
        single_image_seconds_excluding_model_load + model_load_seconds + warmup_seconds
    )

    timing = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "torchvision_weights": args.weights,
        "loaded_checkpoint": loaded_checkpoint,
        "six_class_head_initialized": "checkpoint" if loaded_checkpoint else "random",
        "image_width": width,
        "image_height": height,
        "detected_sift_keypoints": len(keypoints),
        "accepted_nonoverlapping_crops": len(records),
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "model_output": args.model_output,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mixed_precision": args.mixed_precision,
        "compiled": args.compile,
        "preprocessing_seconds": preprocessing_seconds,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_crop": (inference_seconds * 1000.0 / len(records)) if records else None,
        "postprocess_seconds": postprocess_seconds,
        "single_image_seconds_excluding_model_load": single_image_seconds_excluding_model_load,
        "end_to_end_seconds_including_model_load": end_to_end_seconds_including_model_load,
    }
    with (args.out_dir / "timing.json").open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)

    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
