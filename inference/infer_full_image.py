import argparse
import csv
import os

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.utils import extract_peaks, extract_peaks_torch, hann2d, iter_tile_origins, read_prior, read_rgb

import timeit

MULTISCALE_TILE_SIZES = {
    1: 768,
    2: 624,
    4: 432,
    8: 224,
}


def load_model(ckpt_path, device):
    with tqdm(total=3, desc="load model") as pbar:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"]
        pbar.update(1)

        model = DINOv3LesionDetector(
            cfg["model"]["hf_model_name"],
            freeze_backbone=True,
            use_prior=cfg["model"].get("use_prior", True),
            head_channels=cfg["model"].get("head_channels", 256),
        ).to(device)
        pbar.update(1)

        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        pbar.update(1)
    return model, cfg


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return str(device)


def _output_paths(image_path, out_dir):
    prefix = image_path.split('\\')
    prefix = prefix[0].split('/')[-1] + "_" + "_".join(prefix[1:])

    heat_path = os.path.join(out_dir, f"{prefix}_lesion_heatmap.npy")
    png_path = os.path.join(out_dir, f"{prefix}_lesion_heatmap.png")
    csv_path = os.path.join(out_dir, f"{prefix}_candidates.csv")
    return heat_path, png_path, csv_path


def _pad_tile(tile_arr, tile_size):
    h, w = tile_arr.shape[:2]
    if h == tile_size and w == tile_size:
        return tile_arr

    border_type = cv2.BORDER_REFLECT if h > 1 and w > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(
        tile_arr,
        0,
        tile_size - h,
        0,
        tile_size - w,
        border_type,
    )


def _infer_probability_map(image_path, prior, model, cfg, tile, stride, desc_prefix=""):
    device = next(model.parameters()).device
    img = read_rgb(image_path)
    h, w = img.shape[:2]
    if prior is None:
        prior = np.zeros((h, w), dtype=np.float32)
    elif prior.shape[:2] != (h, w):
        prior = cv2.resize(prior, (w, h), interpolation=cv2.INTER_LINEAR)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)

    bs = int(cfg["infer"].get("batch_size", 4))
    alpha = float(cfg["infer"].get("prior_blend_alpha", 0.0))

    score = torch.zeros((h, w), device=device, dtype=torch.float32)
    weight = torch.zeros((h, w), device=device, dtype=torch.float32)
    win = torch.from_numpy(hann2d(tile)).to(device)

    origins = list(iter_tile_origins(w, h, tile, stride))
    batch_imgs, batch_priors, batch_origins = [], [], []
    tile_pbar = tqdm(total=len(origins), desc=f"{desc_prefix}tiles", unit="tile")
    batch_pbar = tqdm(total=(len(origins) + bs - 1) // bs, desc=f"{desc_prefix}forward batches", unit="batch")

    def flush():
        nonlocal batch_imgs, batch_priors, batch_origins, score, weight
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).float().permute(0, 3, 1, 2) / 255.0
        p = torch.from_numpy(np.stack(batch_priors)[:, None]).float()
        x, p = x.to(device), p.to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(x, p))[:, 0].float()
        for pr, (x0, y0) in zip(prob, batch_origins):
            h_part = min(tile, h - y0)
            w_part = min(tile, w - x0)
            pr = pr[:h_part, :w_part]
            if alpha > 0:
                prior_tile = torch.from_numpy(prior[y0:y0+h_part, x0:x0+w_part]).to(device)
                pr = (1 - alpha) * pr + alpha * prior_tile
            score[y0:y0+h_part, x0:x0+w_part] += pr * win[:h_part, :w_part]
            weight[y0:y0+h_part, x0:x0+w_part] += win[:h_part, :w_part]
        batch_pbar.update(1)
        batch_imgs, batch_priors, batch_origins = [], [], []

    for x0, y0 in origins:
        tile_img = img[y0:y0+tile, x0:x0+tile]
        tile_prior = prior[y0:y0+tile, x0:x0+tile]
        batch_imgs.append(_pad_tile(tile_img, tile))
        batch_priors.append(_pad_tile(tile_prior, tile))
        batch_origins.append((x0, y0))
        tile_pbar.update(1)
        if len(batch_imgs) >= bs:
            flush()
    flush()
    tile_pbar.close()
    batch_pbar.close()

    tqdm.write("stitching tiles")
    full_prob_gpu = score / torch.clamp(weight, min=1e-6)
    return full_prob_gpu


def _save_inference_outputs(image_path, out_dir, full_prob_gpu, cfg):
    full_prob = full_prob_gpu.cpu().numpy()
    tqdm.write("extracting peaks")
    pts, scores = extract_peaks_torch(
        full_prob_gpu,
        threshold=float(cfg["infer"]["threshold"]),
        radius=int(cfg["infer"]["nms_radius_px"]),
        max_candidates=int(cfg["infer"].get("max_candidates", 5000)),
    )

    heat_path, png_path, csv_path = _output_paths(image_path, out_dir)
    with tqdm(total=3, desc="save outputs") as pbar:
        np.save(heat_path, full_prob)
        pbar.update(1)
        cv2.imwrite(png_path, (np.clip(full_prob, 0, 1) * 255).astype(np.uint8))
        pbar.update(1)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "score"])
            for (x, y), s in zip(pts, scores):
                writer.writerow([float(x), float(y), float(s)])
        pbar.update(1)
    return heat_path, png_path, csv_path, len(pts)


def run_inference(image_path, prior_path, model, cfg, out_dir):
    t0 = timeit.default_timer()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    print("inference on:", _device_name(device))
    tqdm.write("reading image/prior")
    img = read_rgb(image_path)
    h, w = img.shape[:2]
    prior = read_prior(prior_path, (h, w))

    tile = int(cfg["infer"]["tile_size"])
    stride = int(cfg["infer"]["stride"])

    full_prob_gpu = _infer_probability_map(image_path, prior, model, cfg, tile, stride)
    t1 = timeit.default_timer()
    heat_path, png_path, csv_path, num_candidates = _save_inference_outputs(image_path, out_dir, full_prob_gpu, cfg)

    
    print(f"Inference complete in {t1-t0:.2f} seconds. Found {num_candidates} candidates.")

    return heat_path, png_path, csv_path


def _multiscale_image_paths(image_path_1x):
    normalized = image_path_1x.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive multiscale paths from image path: {image_path_1x}")

    patient = parts[-3]
    postfix = "/".join(parts[-2:])
    prefix = "/".join(parts[:-3])

    return {
        1: image_path_1x,
        2: prefix + f"/{patient}_downsample_2x/" + postfix,
        4: prefix + f"/{patient}_downsample_4x/" + postfix,
        8: prefix + f"/{patient}_downsample_8x/" + postfix,
    }


def run_multiscale_inference(image_path_1x, model, cfg, out_dir):
    t0 = timeit.default_timer()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    print("multiscale inference on:", _device_name(device))

    image_paths = _multiscale_image_paths(image_path_1x)
    missing = [path for path in image_paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing multiscale image path(s): " + ", ".join(missing))

    full_tile = int(cfg["infer"]["tile_size"])
    full_stride = int(cfg["infer"]["stride"])
    prior = None

    for scale in [8, 4, 2, 1]:
        image_path = image_paths[scale]
        tile = int(MULTISCALE_TILE_SIZES.get(scale, full_tile))
        stride = max(1, int(round(full_stride * tile / full_tile)))
        tqdm.write(f"running {scale}x inference: tile={tile}, stride={stride}")

        full_prob_gpu = _infer_probability_map(
            image_path,
            prior,
            model,
            cfg,
            tile,
            stride,
            desc_prefix=f"{scale}x ",
        )
        prior = full_prob_gpu.cpu().numpy()
    t1 = timeit.default_timer()

    heat_path, png_path, csv_path, num_candidates = _save_inference_outputs(
        image_path_1x,
        out_dir,
        full_prob_gpu,
        cfg,
    )

    
    print(f"Multiscale inference complete in {t1-t0:.2f} seconds. Found {num_candidates} candidates.")
    return heat_path, png_path, csv_path




def main(chk = None, img = None, fac = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)#
    ap.add_argument("--image", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--multiscale", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chkpnt = args.checkpoint if chk == None else chk
    image = args.image if img == None else img
    print("using device:", device, _device_name(device))
    model, cfg = load_model(chkpnt, device)
    print("model loaded. Starting inference...")
    out = f"output/{fac}" if fac != None else args.out_dir
    if args.multiscale:
        heat_path, png_path, csv_path = run_multiscale_inference(image, model, cfg, out)
    else:
        heat_path, png_path, csv_path = run_inference(image, args.prior, model, cfg, out)

    print("saved", heat_path, png_path, csv_path)
    if chk or img:
        return heat_path, png_path, csv_path

if __name__ == "__main__":
    main()
