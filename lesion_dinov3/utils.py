import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.diameter_model import DINOv3LesionDiameterDetector

Point = Tuple[float, float] | Tuple[float, float, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(ckpt_path, device, diameter = False):
    with tqdm(total=3, desc="load model") as pbar:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"]
        pbar.update(1)

        if diameter:
            model = DINOv3LesionDiameterDetector(
                cfg["model"]["hf_model_name"],
                freeze_backbone=True,
                use_prior=cfg["model"].get("use_prior", True),
                head_channels=cfg["model"].get("head_channels", 256),
            ).to(device)
        else:
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

def parse_points(points_json: str) -> List[Point]:
    if points_json is None or str(points_json).strip() == "" or str(points_json).lower() == "nan":
        return []
    pts = json.loads(points_json)
    try:
        return [(float(x), float(y), float(d)) for x, y, d in pts]
    except ValueError as e:
        return [(float(x), float(y), 1) for x, y in pts]


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_prior(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    if path is None or str(path).strip() == "" or str(path).lower() == "nan" or not os.path.exists(path):
        return np.zeros(shape_hw, dtype=np.float32)
    p = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if p is None:
        raise FileNotFoundError(path)
    if p.ndim == 3:
        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    p = p.astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    if p.shape[:2] != shape_hw:
        p = cv2.resize(p, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(p, 0.0, 1.0)


def make_heatmap(h: int, w: int, points: Sequence[Point], sigma: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    hm = np.zeros((h, w), dtype=np.float32)
    for point in points:
        x, y = point[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        hm = np.maximum(hm, g)
    return hm

def make_diameter_aware_heatmap(h: int, w: int, points: Sequence[Point], image_scale: int = 1, min_sigma: float = 3.0, max_sigma: float = 12) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    hm = np.zeros((h, w), dtype=np.float32)

    # pbar = tqdm.tqdm(points, desc="Generating heatmap", unit="point")
    for x, y, d in points:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        radius = d * 0.5
        sigma_1x = np.clip(0.25 * radius, min_sigma, max_sigma) # base sigma at original image scale
        sigma_s = max(1.0, sigma_1x / image_scale) # sigma scaled to output resolution

        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_s ** 2)).astype(np.float32)
        hm = np.maximum(hm, g)
    return hm


def points_in_tile(points: Sequence[Point], x0: int, y0: int, tile_size: int) -> List[Point]:
    out = []
    x1, y1 = x0 + tile_size, y0 + tile_size
    if len(points) == 0:
        return out
    if len(points[0]) == 3:
        for x, y, d in points:
            if x0 <= x < x1 and y0 <= y < y1:
                out.append((x - x0, y - y0, d))
    else:
        for x, y in points:
            if x0 <= x < x1 and y0 <= y < y1:
                out.append((x - x0, y - y0))
    return out


def valid_tile_origin(x0: int, y0: int, w: int, h: int, tile: int) -> Tuple[int, int]:
    x0 = int(np.clip(x0, 0, max(0, w - tile)))
    y0 = int(np.clip(y0, 0, max(0, h - tile)))
    return x0, y0


def iter_tile_origins(w: int, h: int, tile: int, stride: int):
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if len(xs) == 0 or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    if len(ys) == 0 or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def nms_points(points: np.ndarray, scores: np.ndarray, radius: float, max_keep: int = 5000):
    if len(points) == 0:
        return points, scores
    order = np.argsort(scores)[::-1]
    keep = []
    r2 = radius * radius
    for idx in order:
        p = points[idx]
        ok = True
        for kept_idx in keep:
            d = p - points[kept_idx]
            if float(d[0] * d[0] + d[1] * d[1]) <= r2:
                ok = False
                break
        if ok:
            keep.append(idx)
            if len(keep) >= max_keep:
                break
    return points[keep], scores[keep]


def extract_peaks(prob: np.ndarray, threshold: float, radius: int, max_candidates: int):
    t = torch.from_numpy(prob[None, None].astype(np.float32))
    k = 2 * radius + 1
    pooled = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    mask = (t == pooled) & (t >= threshold)
    ys, xs = torch.where(mask[0, 0])
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    scores = prob[ys.numpy(), xs.numpy()]
    pts = np.stack([xs.numpy(), ys.numpy()], axis=1).astype(np.float32)
    pts, scores = nms_points(pts, scores, radius=radius, max_keep=max_candidates)
    return pts, scores.astype(np.float32)

def extract_peaks_torch(
    prob: torch.Tensor,
    threshold: float,
    radius: int,
    max_candidates: int,
):
    """
    prob: [H, W] CUDA tensor
    returns CPU numpy pts, scores
    """
    if prob.ndim != 2:
        raise ValueError("prob must be [H, W]")

    t = prob[None, None].float()
    k = 2 * radius + 1

    pooled = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    mask = (t == pooled) & (t >= threshold)

    ys, xs = torch.where(mask[0, 0])
    if xs.numel() == 0:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
        )

    scores = prob[ys, xs]

    # Keep only top candidates before CPU NMS.
    if scores.numel() > max_candidates * 10:
        keep_n = max_candidates * 10
        scores, idx = torch.topk(scores, keep_n)
        xs = xs[idx]
        ys = ys[idx]

    pts = torch.stack([xs, ys], dim=1).float()

    # Move only compact candidate list to CPU, not full heatmap.
    pts_np = pts.cpu().numpy().astype(np.float32)
    scores_np = scores.cpu().numpy().astype(np.float32)

    np.save("debug_pts.npy", pts_np)
    np.save("debug_scores.npy", scores_np)

    pts_np, scores_np = nms_points(
        pts_np,
        scores_np,
        radius=radius,
        max_keep=max_candidates,
    )
    return pts_np, scores_np

def hann2d(tile_size: int) -> np.ndarray:
    w = np.hanning(tile_size).astype(np.float32)
    win = np.outer(w, w).astype(np.float32)
    win = np.maximum(win, 1e-3)
    return win


def format_duration(seconds: int) -> str:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}"
