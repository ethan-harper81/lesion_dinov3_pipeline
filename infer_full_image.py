import argparse
import csv
import os

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.utils import extract_peaks, hann2d, iter_tile_origins, read_prior, read_rgb


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


def run_inference(image_path, prior_path, model, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    print("inference on:", torch.cuda.get_device_name(0))
    tqdm.write("reading image/prior")
    img = read_rgb(image_path)
    h, w = img.shape[:2]
    prior = read_prior(prior_path, (h, w))

    tile = int(cfg["infer"]["tile_size"])
    stride = int(cfg["infer"]["stride"])
    bs = int(cfg["infer"].get("batch_size", 4))
    alpha = float(cfg["infer"].get("prior_blend_alpha", 0.0))

    score = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    win = hann2d(tile)

    origins = list(iter_tile_origins(w, h, tile, stride))
    batch_imgs, batch_priors, batch_origins = [], [], []
    tile_pbar = tqdm(total=len(origins), desc="tiles", unit="tile")
    batch_pbar = tqdm(total=(len(origins) + bs - 1) // bs, desc="forward batches", unit="batch")

    def flush():
        nonlocal batch_imgs, batch_priors, batch_origins, score, weight
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).float().permute(0, 3, 1, 2) / 255.0
        p = torch.from_numpy(np.stack(batch_priors)[:, None]).float()
        x, p = x.to(device), p.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(x, p)).detach().cpu().numpy()[:, 0]
        for pr, (x0, y0) in zip(prob, batch_origins):
            if alpha > 0:
                prior_tile = prior[y0:y0+tile, x0:x0+tile]
                pr = (1 - alpha) * pr + alpha * prior_tile
            score[y0:y0+tile, x0:x0+tile] += pr * win
            weight[y0:y0+tile, x0:x0+tile] += win
        batch_pbar.update(1)
        batch_imgs, batch_priors, batch_origins = [], [], []

    for x0, y0 in origins:
        tile_img = img[y0:y0+tile, x0:x0+tile]
        tile_prior = prior[y0:y0+tile, x0:x0+tile]
        # Origins from iter_tile_origins avoid partial tiles for normal full-res images.
        batch_imgs.append(tile_img)
        batch_priors.append(tile_prior)
        batch_origins.append((x0, y0))
        tile_pbar.update(1)
        if len(batch_imgs) >= bs:
            flush()
    flush()
    tile_pbar.close()
    batch_pbar.close()

    tqdm.write("stitching tiles")
    full_prob = score / np.maximum(weight, 1e-6)
    tqdm.write("extracting peaks")
    pts, scores = extract_peaks(
        full_prob,
        threshold=float(cfg["infer"]["threshold"]),
        radius=int(cfg["infer"]["nms_radius_px"]),
        max_candidates=int(cfg["infer"].get("max_candidates", 5000)),
    )

    stem = os.path.splitext(os.path.basename(image_path))[0]
    heat_path = os.path.join(out_dir, f"{stem}_lesion_heatmap.npy")
    png_path = os.path.join(out_dir, f"{stem}_lesion_heatmap.png")
    csv_path = os.path.join(out_dir, f"{stem}_candidates.csv")
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
    return heat_path, png_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    heat_path, png_path, csv_path = run_inference(args.image, args.prior, model, cfg, args.out_dir)
    print("saved", heat_path, png_path, csv_path)


if __name__ == "__main__":
    main()
