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
        self.val_sampling = cfg["data"].get("validation_sampling", {})
        self.diameter_aware = bool(cfg["data"].get("diameter_aware_heatmap", {}).get("enabled", False))
        
        self.records = []
        for _, row in self.df.iterrows():
            
            pts = parse_points(row.get("points_json", "[]"))
            self.records.append({
                "image_path": row["image_path"],
                "points": pts,
                # "points_diameter": pts_diameter if self.diameter_aware else None,
                "prior_path": row.get("prior_path", None),
            })
        
        self.avg_diameter = np.mean([p[2] for r in self.records for p in r["points"]]) if self.diameter_aware else None

        self.pos_indices = [i for i, r in enumerate(self.records) if len(r["points"]) > 0]
        self.all_indices = list(range(len(self.records)))
        self.val_tiles = self._build_validation_tiles() if (not self.training and self.val_sampling.get("enabled", False)) else None

    def __len__(self):
        if self.training:
            return self.samples_per_epoch
        if self.val_tiles is not None:
            return len(self.val_tiles)
        return len(self.records)

    def _sample_record(self):
        if self.training and random.random() < float(self.cfg["data"].get("positive_fraction", 0.4)) and self.pos_indices:
            return self.records[random.choice(self.pos_indices)], "positive"
        return self.records[random.choice(self.all_indices)], "mixed"

    def _choose_tile_from_shape_prior(self, w, h, prior, points, mode, rng=random):
        tile = self.tile
        if mode == "positive" and len(points) > 0:
            x, y, _ = rng.choice(points)
            jitter = tile // 3
            x0 = int(x - tile / 2 + rng.randint(-jitter, jitter))
            y0 = int(y - tile / 2 + rng.randint(-jitter, jitter))
            return valid_tile_origin(x0, y0, w, h, tile)

        r = rng.random()
        high_prior_frac = float(self.cfg["data"].get("high_prior_negative_fraction", 0.3))
        low_prior_frac = float(self.cfg["data"].get("low_prior_fraction", 0.1))
        threshold = float(self.cfg["data"].get("prior_threshold", 0.35))

        if mode == "high_prior" or (mode == "mixed" and r < high_prior_frac):
            if prior.max() <= threshold:
                return rng.randint(0, max(0, w - tile)), rng.randint(0, max(0, h - tile))
            ys, xs = np.where(prior >= threshold)
            if len(xs) > 0:
                j = rng.randrange(len(xs))
                return valid_tile_origin(int(xs[j] - tile / 2), int(ys[j] - tile / 2), w, h, tile)

        if mode == "low_prior" or (mode == "mixed" and r < high_prior_frac + low_prior_frac and prior.size > 0):
            # Prefer low-prior/background, but fall back to random.
            for _ in range(20):
                x0 = rng.randint(0, max(0, w - tile))
                y0 = rng.randint(0, max(0, h - tile))
                if prior[y0:y0+tile, x0:x0+tile].mean() < threshold * 0.25:
                    return x0, y0

        return rng.randint(0, max(0, w - tile)), rng.randint(0, max(0, h - tile))

    def _choose_tile(self, img, prior, points, mode):
        h, w = img.shape[:2]
        return self._choose_tile_from_shape_prior(w, h, prior, points, mode)

    def _build_validation_tiles(self):
        samples_per_image = int(self.val_sampling.get("samples_per_image", 8))
        seed = int(self.val_sampling.get("seed", self.cfg.get("seed", 1337)))
        positive_fraction = float(self.val_sampling.get("positive_fraction", self.cfg["data"].get("positive_fraction", 0.4)))
        high_prior_fraction = float(self.val_sampling.get("high_prior_fraction", self.cfg["data"].get("high_prior_negative_fraction", 0.3)))
        low_prior_fraction = float(self.val_sampling.get("low_prior_fraction", self.cfg["data"].get("low_prior_fraction", 0.1)))
        rng = random.Random(seed)

        positive_count = round(samples_per_image * positive_fraction)
        high_prior_count = round(samples_per_image * high_prior_fraction)
        low_prior_count = round(samples_per_image * low_prior_fraction)
        random_count = max(0, samples_per_image - positive_count - high_prior_count - low_prior_count)
        modes = (
            ["positive"] * positive_count
            + ["high_prior"] * high_prior_count
            + ["low_prior"] * low_prior_count
            + ["random"] * random_count
        )
        modes = modes[:samples_per_image]

        tiles = []
        for rec_idx, rec in enumerate(self.records):
            img = read_rgb(rec["image_path"])
            h, w = img.shape[:2]
            prior = read_prior(rec.get("prior_path"), (h, w))

            record_modes = modes[:]
            rng.shuffle(record_modes)
            for mode in record_modes:
                if mode == "positive" and not rec["points"]:
                    mode = "random"
                x0, y0 = self._choose_tile_from_shape_prior(w, h, prior, rec["points"], mode, rng)
                tiles.append({"record_idx": rec_idx, "origin": (x0, y0), "mode": mode})
        return tiles

    def __getitem__(self, idx):
        if self.training:
            rec, mode = self._sample_record()
            fixed_origin = None
        elif self.val_tiles is not None:
            tile = self.val_tiles[idx]
            rec, mode = self.records[tile["record_idx"]], tile["mode"]
            fixed_origin = tile["origin"]
        else:
            rec, mode = self.records[idx], "center"
            fixed_origin = None

        img = read_rgb(rec["image_path"])
        h, w = img.shape[:2]
        prior = read_prior(rec.get("prior_path"), (h, w))

        if self.training:
            x0, y0 = self._choose_tile(img, prior, rec["points"], mode)
        elif fixed_origin is not None:
            x0, y0 = fixed_origin
        else:
            # Validation samples one tile centered around a lesion if possible, otherwise center image.
            if rec["points"]:
                x, y, _ = rec["points"][0]
                x0, y0 = valid_tile_origin(int(x - self.tile / 2), int(y - self.tile / 2), w, h, self.tile)
            else:
                x0, y0 = valid_tile_origin((w - self.tile) // 2, (h - self.tile) // 2, w, h, self.tile)

        tile_img = img[y0:y0+self.tile, x0:x0+self.tile]
        tile_prior = prior[y0:y0+self.tile, x0:x0+self.tile]

        # Pad edge cases for images smaller than tile.
        if tile_img.shape[0] != self.tile or tile_img.shape[1] != self.tile:
            tile_img = cv2.copyMakeBorder(tile_img, 0, self.tile - tile_img.shape[0], 0, self.tile - tile_img.shape[1], cv2.BORDER_REFLECT)
            tile_prior = cv2.copyMakeBorder(tile_prior, 0, self.tile - tile_prior.shape[0], 0, self.tile - tile_prior.shape[1], cv2.BORDER_REFLECT)
        local_pts = points_in_tile(rec["points"], x0, y0, self.tile)
        if self.diameter_aware:
            target = make_diameter_aware_heatmap(
                self.tile, self.tile, local_pts, 
                image_scale=1, 

                min_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("min_sigma", 3.0), 
                max_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("max_sigma", 12.0)
            )
        else:
            target = make_heatmap(self.tile, self.tile, local_pts, self.sigma)

        x = torch.from_numpy(tile_img).float().permute(2, 0, 1) / 255.0
        p = torch.from_numpy(tile_prior[None]).float()
        y = torch.from_numpy(target[None]).float()
        return {"image": x, "prior": p, "target": y, "origin": torch.tensor([x0, y0]), "num_points": len(local_pts)}

class MultiScaleLesionTileDataset(Dataset):
    """
    
    CSV columns:
      image_path_1: path to full-image resolution RGB image
      image_path_2: path to 2x resolution RGB image
      image_path_4: path to 4x resolution RGB image
      image_path_8: path to 8x resolution RGB image
      pts_json: JSON list [[x,y], ...] in full-image coordinates

    """

    def __init__(self, csv_path: str, cfg: Dict, training: bool = True):
        self.df = pd.read_csv(csv_path)

        self.cfg = cfg
        self.training = training
        self.tile = int(cfg["data"]["tile_size"])
        self.sigma = float(cfg["data"]["sigma_px"])
        self.samples_per_epoch = int(cfg["data"].get("train_samples_per_epoch", len(self.df)))
        self.val_sampling = cfg["data"].get("validation_sampling", {})
        self.diameter_aware = bool(cfg["data"].get("diameter_aware_heatmap", {}).get("enabled", False))

        self.tile_sizes = [768, 624, 432, 224]
        self.scales = [1, 2, 4, 8]

        self.records = []

        column_names = self.df.columns.to_list()
        pts_columns = [col for col in column_names if "points" in col]
        path_columns = [col for col in column_names if "path" in col]
        pts_columns.sort()
        path_columns.sort()
        points_column = "points_json" if "points_json" in column_names else "pts_json"

        for _, row in self.df.iterrows():
            self.records.append({
                "image_path_1": row["image_path_1"],
                "image_path_2": row["image_path_2"],
                "image_path_4": row["image_path_4"],
                "image_path_8": row["image_path_8"],
                "points": parse_points(row.get(points_column, "[]")),

            })
        
        #Full res diameter
        self.avg_diameter = np.mean([p[2] for r in self.records for p in r["points"]]) if self.diameter_aware else None

        self.pos_indices = [i for i, r in enumerate(self.records) if len(r["points"]) > 0]
        self.all_indices = list(range(len(self.records)))
        self.val_tiles = None
        
    def __len__(self):
        if self.training:
            return self.samples_per_epoch * 4
        if self.val_tiles is not None:
            return len(self.val_tiles)
        return len(self.records)

    def _sample_record(self):
        """
        returns a single record
        """
        if self.training and random.random() < float(self.cfg["data"].get("positive_fraction", 0.4)) and self.pos_indices:
            return self.records[random.choice(self.pos_indices)], "positive"
        return self.records[random.choice(self.all_indices)], "mixed"
        
    def _choose_tile(self, img, points, mode, rng=random):
        h,w = img.shape[:2]

        tile = self.tile_sizes[0]
        if mode == "positive" and len(points) > 0:
            x,y, _ = rng.choice(points)
            jitter = tile //3
            x0 = int(x - tile / 2 + rng.randint(-jitter, jitter))
            y0 = int(y - tile / 2 + rng.randint(-jitter, jitter))
            return valid_tile_origin(x0, y0, w, h, tile)
        
        return rng.randint(0, max(0, w - tile)), rng.randint(0, max(0, h - tile))

    def _points_in_scaled_tile(self, points, x0, y0, tile_size, scale):
        local = []
        x1 = x0 + tile_size * scale
        y1 = y0 + tile_size * scale
        for x, y, d in points:
            if x0 <= x < x1 and y0 <= y < y1:
                local.append(((x - x0) / scale, (y - y0) / scale, d))
        return local
        
    def __getitem__(self, idx):
        if self.training:
            rec, mode = self._sample_record()
            fixed_origin = None
        
        imgs = [read_rgb(rec["image_path_1"]),read_rgb(rec["image_path_2"]), read_rgb(rec["image_path_4"]),read_rgb(rec["image_path_8"])]

        if self.training:
            x0,y0 = self._choose_tile(imgs[0], rec['points'], mode)
        
        tiles = [
            img[
                y0 // s: y0 // s + tile_size, 
                x0 // s: x0 // s + tile_size, 
            ] for img, tile_size, s in zip(imgs, self.tile_sizes, self.scales)
        ]

        for i in range(len(tiles)):
            expected = self.tile_sizes[i]
            h, w = tiles[i].shape[:2]

            if h != expected or w != expected:
                tiles[i] = cv2.copyMakeBorder(
                    tiles[i],
                    0, expected - h,
                    0, expected - w,
                    cv2.BORDER_REFLECT
                )
        local_pts_by_scale = [
            self._points_in_scaled_tile(rec["points"], x0, y0, tile, scale)
            for tile, scale in zip(self.tile_sizes, self.scales)
        ]
        local_pts = local_pts_by_scale[0]
        targets = []
        if self.diameter_aware:
            targets = [make_diameter_aware_heatmap(
                tile, tile, pts,
                image_scale=scale,
                min_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("min_sigma", 3.0), 
                max_sigma=self.cfg["data"]["diameter_aware_heatmap"].get("max_sigma", 12.0 / scale)
            ) for tile, scale, pts in zip(self.tile_sizes, self.scales, local_pts_by_scale)
            ]
        else:
            targets = [
                make_heatmap(tile, tile, pts, max(1.0, self.sigma / scale))
                for tile, scale, pts in zip(self.tile_sizes, self.scales, local_pts_by_scale)
            ]

        xs = [torch.from_numpy(tile).float().permute(2,0,1) / 255.0 for tile in tiles]
        ys = [torch.from_numpy(target[None]).float() for target in targets]

        return {
            "images": xs,
            "targets": ys,
            "origins": [torch.tensor([x0 // scale,y0 //scale]) for scale in self.scales],
            "num_points": len(local_pts)
        }
