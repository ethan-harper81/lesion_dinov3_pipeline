import math
import random
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import make_diameter_aware_heatmap, make_heatmap, parse_points, points_in_tile, read_prior, read_rgb, valid_tile_origin


class LesionTileDataset(Dataset):
    """Samples native-resolution tiles from huge images with point labels.

    CSV columns:
      image_path: path to RGB image
      points_json: JSON list [[x,y], ...] in full-image coordinates
      prior_path: optional path to grayscale prior heatmap image
    """

    def __init__(self, csv_path: str, cfg: Dict, training: bool = True):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.training = training
        self.tile = int(cfg["data"]["tile_size"])
        self.sigma = float(cfg["data"]["sigma_px"])
        self.samples_per_epoch = int(cfg["data"].get("train_samples_per_epoch", len(self.df)))
        self.diameter_aware = bool(cfg["data"].get("diameter_aware_heatmap", {}).get("enabled", False))
        #rec['points'] is list of (x,y) tuples
        #rec['points_diameter'] is list of (x,y,diameter) tuples
        self.records = []
        for _, row in self.df.iterrows():
            if self.diameter_aware:
                pts, pts_diameter = parse_points(row.get("points_json", "[]"), diameter_aware=True)
            pts = parse_points(row.get("points_json", "[]"), diameter_aware=self.diameter_aware)
            self.records.append({
                "image_path": row["image_path"],
                "points": pts,
                "points_diameter": pts_diameter if self.diameter_aware else None,
                "prior_path": row.get("prior_path", None),
            })

        self.pos_indices = [i for i, r in enumerate(self.records) if len(r["points"]) > 0]
        self.all_indices = list(range(len(self.records)))

    def __len__(self):
        return self.samples_per_epoch if self.training else len(self.records)

    def _sample_record(self):
        if self.training and random.random() < float(self.cfg["data"].get("positive_fraction", 0.4)) and self.pos_indices:
            return self.records[random.choice(self.pos_indices)], "positive"
        return self.records[random.choice(self.all_indices)], "mixed"

    def _choose_tile(self, img, prior, points, mode):
        h, w = img.shape[:2]
        tile = self.tile
        if mode == "positive" and len(points) > 0:
            x, y = random.choice(points)
            jitter = tile // 3
            x0 = int(x - tile / 2 + random.randint(-jitter, jitter))
            y0 = int(y - tile / 2 + random.randint(-jitter, jitter))
            return valid_tile_origin(x0, y0, w, h, tile)

        r = random.random()
        high_prior_frac = float(self.cfg["data"].get("high_prior_negative_fraction", 0.3))
        low_prior_frac = float(self.cfg["data"].get("low_prior_fraction", 0.1))
        threshold = float(self.cfg["data"].get("prior_threshold", 0.35))

        if r < high_prior_frac and prior.max() > threshold:
            ys, xs = np.where(prior >= threshold)
            if len(xs) > 0:
                j = random.randrange(len(xs))
                return valid_tile_origin(int(xs[j] - tile / 2), int(ys[j] - tile / 2), w, h, tile)

        if r < high_prior_frac + low_prior_frac and prior.size > 0:
            # Prefer low-prior/background, but fall back to random.
            for _ in range(20):
                x0 = random.randint(0, max(0, w - tile))
                y0 = random.randint(0, max(0, h - tile))
                if prior[y0:y0+tile, x0:x0+tile].mean() < threshold * 0.25:
                    return x0, y0

        return random.randint(0, max(0, w - tile)), random.randint(0, max(0, h - tile))

    def __getitem__(self, idx):
        if self.training:
            rec, mode = self._sample_record()
        else:
            rec, mode = self.records[idx], "center"

        img = read_rgb(rec["image_path"])
        h, w = img.shape[:2]
        prior = read_prior(rec.get("prior_path"), (h, w))

        if self.training:
            x0, y0 = self._choose_tile(img, prior, rec["points"], mode)
        else:
            # Validation samples one tile centered around a lesion if possible, otherwise center image.
            if rec["points"]:
                x, y = rec["points"][0]
                x0, y0 = valid_tile_origin(int(x - self.tile / 2), int(y - self.tile / 2), w, h, self.tile)
            else:
                x0, y0 = valid_tile_origin((w - self.tile) // 2, (h - self.tile) // 2, w, h, self.tile)

        tile_img = img[y0:y0+self.tile, x0:x0+self.tile]
        tile_prior = prior[y0:y0+self.tile, x0:x0+self.tile]

        # Pad edge cases for images smaller than tile.
        if tile_img.shape[0] != self.tile or tile_img.shape[1] != self.tile:
            tile_img = cv2.copyMakeBorder(tile_img, 0, self.tile - tile_img.shape[0], 0, self.tile - tile_img.shape[1], cv2.BORDER_REFLECT)
            tile_prior = cv2.copyMakeBorder(tile_prior, 0, self.tile - tile_prior.shape[0], 0, self.tile - tile_prior.shape[1], cv2.BORDER_REFLECT)

        if self.diameter_aware:
            local_pts = points_in_tile(rec["points_diameter"], x0, y0, self.tile, diameter_aware=True)
            target = make_diameter_aware_heatmap(self.tile, self.tile, local_pts, image_scale=1, sigma_factor=self.cfg["data"]["diameter_aware_heatmap"].get("sigma_factor", 25), min_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("min_sigma", 1.0), max_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("max_sigma", 6))
        else:
            local_pts = points_in_tile(rec["points"], x0, y0, self.tile)
        
            target = make_heatmap(self.tile, self.tile, local_pts, self.sigma)

        x = torch.from_numpy(tile_img).float().permute(2, 0, 1) / 255.0
        p = torch.from_numpy(tile_prior[None]).float()
        y = torch.from_numpy(target[None]).float()
        return {"image": x, "prior": p, "target": y, "origin": torch.tensor([x0, y0]), "num_points": len(local_pts)}
