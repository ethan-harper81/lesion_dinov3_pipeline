import argparse
import csv
import os
import timeit

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from lesion_dinov3.utils import (
    extract_peaks_torch, 
    hann2d, 
    iter_tile_origins, 
    read_prior, 
    read_rgb, 
    load_model
)

def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return str(device)


def _output_paths(image_path, out_dir):
    prefix = image_path.split("\\")
    prefix = prefix[0].split("/")[-1] + "_" + "_".join(prefix[1:])
    heat_path = os.path.join(out_dir, f"{prefix}_lesion_heatmap.npy")
    png_path = os.path.join(out_dir, f"{prefix}_lesion_heatmap.png")
    diameter_path = os.path.join(out_dir, f"{prefix}_log_diameter.npy")
    csv_path = os.path.join(out_dir, f"{prefix}_candidates.csv")
    return heat_path, png_path, diameter_path, csv_path


def _pad_tile(tile_arr, tile_size):
    h, w = tile_arr.shape[:2]
    if h == tile_size and w == tile_size:
        return tile_arr
    border_type = cv2.BORDER_REFLECT if h > 1 and w > 1 else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(tile_arr, 0, tile_size - h, 0, tile_size - w, border_type)


def infer_maps(image_path, prior, model, cfg, tile, stride):
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
    log_diameter_sum = torch.zeros((h, w), device=device, dtype=torch.float32)
    weight = torch.zeros((h, w), device=device, dtype=torch.float32)
    win = torch.from_numpy(hann2d(tile)).to(device)

    origins = list(iter_tile_origins(w, h, tile, stride))
    batch_imgs, batch_priors, batch_origins = [], [], []
    tile_pbar = tqdm(total=len(origins), desc="tiles", unit="tile")
    batch_pbar = tqdm(total=(len(origins) + bs - 1) // bs, desc="forward batches", unit="batch")

    def flush():
        nonlocal batch_imgs, batch_priors, batch_origins
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).float().permute(0, 3, 1, 2) / 255.0
        p = torch.from_numpy(np.stack(batch_priors)[:, None]).float()
        x, p = x.to(device), p.to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            center_logits, log_diameter = model(x, p)
            prob = torch.sigmoid(center_logits)[:, 0].float()
            log_diameter = log_diameter[:, 0].float()

        for pr, ld, (x0, y0) in zip(prob, log_diameter, batch_origins):
            h_part = min(tile, h - y0)
            w_part = min(tile, w - x0)
            pr = pr[:h_part, :w_part]
            ld = ld[:h_part, :w_part]
            tile_win = win[:h_part, :w_part]
            if alpha > 0:
                prior_tile = torch.from_numpy(prior[y0:y0 + h_part, x0:x0 + w_part]).to(device)
                pr = (1 - alpha) * pr + alpha * prior_tile
            score[y0:y0 + h_part, x0:x0 + w_part] += pr * tile_win
            log_diameter_sum[y0:y0 + h_part, x0:x0 + w_part] += ld * tile_win
            weight[y0:y0 + h_part, x0:x0 + w_part] += tile_win

        batch_pbar.update(1)
        batch_imgs, batch_priors, batch_origins = [], [], []

    for x0, y0 in origins:
        tile_img = img[y0:y0 + tile, x0:x0 + tile]
        tile_prior = prior[y0:y0 + tile, x0:x0 + tile]
        batch_imgs.append(_pad_tile(tile_img, tile))
        batch_priors.append(_pad_tile(tile_prior, tile))
        batch_origins.append((x0, y0))
        tile_pbar.update(1)
        if len(batch_imgs) >= bs:
            flush()
    flush()
    tile_pbar.close()
    batch_pbar.close()

    weight = torch.clamp(weight, min=1e-6)
    return score / weight, log_diameter_sum / weight


def save_outputs(image_path, out_dir, full_prob_gpu, full_log_diameter_gpu, cfg):
    full_prob = full_prob_gpu.cpu().numpy()
    full_log_diameter = full_log_diameter_gpu.cpu().numpy()

    tqdm.write("extracting peaks")
    pts, scores = extract_peaks_torch(
        full_prob_gpu,
        threshold=float(cfg["infer"]["threshold"]),
        radius=int(cfg["infer"]["nms_radius_px"]),
        max_candidates=int(cfg["infer"].get("max_candidates", 5000)),
    )

    heat_path, png_path, diameter_path, csv_path = _output_paths(image_path, out_dir)
    with tqdm(total=4, desc="save outputs") as pbar:
        np.save(heat_path, full_prob)
        pbar.update(1)
        cv2.imwrite(png_path, (np.clip(full_prob, 0, 1) * 255).astype(np.uint8))
        pbar.update(1)
        np.save(diameter_path, full_log_diameter)
        pbar.update(1)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "score", "diameter_px"])
            for (x, y), s in zip(pts, scores):
                xi = int(np.clip(round(float(x)), 0, full_log_diameter.shape[1] - 1))
                yi = int(np.clip(round(float(y)), 0, full_log_diameter.shape[0] - 1))
                diameter_px = float(np.exp(full_log_diameter[yi, xi]))
                writer.writerow([float(x), float(y), float(s), diameter_px])
        pbar.update(1)
    return heat_path, png_path, diameter_path, csv_path, len(pts)


def run_inference(image_path, prior_path, model, cfg, out_dir):
    t0 = timeit.default_timer()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    print("diameter inference on:", _device_name(device))

    img = read_rgb(image_path)
    h, w = img.shape[:2]
    prior = read_prior(prior_path, (h, w))
    tile = int(cfg["infer"]["tile_size"])
    stride = int(cfg["infer"]["stride"])

    full_prob_gpu, full_log_diameter_gpu = infer_maps(image_path, prior, model, cfg, tile, stride)
    heat_path, png_path, diameter_path, csv_path, num_candidates = save_outputs(
        image_path,
        out_dir,
        full_prob_gpu,
        full_log_diameter_gpu,
        cfg,
    )
    t1 = timeit.default_timer()
    print(f"Inference complete in {t1 - t0:.2f} seconds. Found {num_candidates} candidates.")
    return heat_path, png_path, diameter_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--out_dir", default="outputs_diameter")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device, _device_name(device))
    model, cfg = load_model(args.checkpoint, device, diameter=True)
    paths = run_inference(args.image, args.prior, model, cfg, args.out_dir)
    print("saved", *paths)


if __name__ == "__main__":
    main()
