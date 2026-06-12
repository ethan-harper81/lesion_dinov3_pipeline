import random
from typing import Dict, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import parse_points, points_in_tile, read_prior, read_rgb, valid_tile_origin


def make_center_heatmap(h: int, w: int, points: Sequence[tuple[float, float, float]], sigma: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    hm = np.zeros((h, w), dtype=np.float32)
    for point in points:
        x, y = point[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        hm = np.maximum(hm, g)
    return hm


def make_log_diameter_targets(
    h: int,
    w: int,
    points: Sequence[tuple[float, float, float]],
    sigma: float,
    mask_threshold: float,
):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    target = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    strength = np.zeros((h, w), dtype=np.float32)

    for x, y, diameter in points:
        if diameter <= 0 or x < 0 or y < 0 or x >= w or y >= h:
            continue
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        point_mask = g >= mask_threshold
        replace = point_mask & (g >= strength)
        target[replace] = np.log(float(diameter))
        mask[replace] = 1.0
        strength[replace] = g[replace]

    return target, mask


class DiameterLesionTileDataset(Dataset):
    """Samples tiles for center heatmap plus log-diameter regression."""

    def __init__(self, csv_path: str, cfg: Dict, training: bool = True):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.training = training
        self.tile = int(cfg["data"]["tile_size"])
        self.sigma = float(cfg["data"].get("sigma_px", 6))
        self.mask_threshold = float(cfg["data"].get("diameter_mask_threshold", 0.25))
        self.samples_per_epoch = int(cfg["data"].get("train_samples_per_epoch", len(self.df)))
        self.val_sampling = cfg["data"].get("validation_sampling", {})

        self.records = []
        for _, row in self.df.iterrows():
            self.records.append(
                {
                    "image_path": row["image_path"],
                    "points": parse_points(row.get("points_json", "[]")),
                    "prior_path": row.get("prior_path", None),
                }
            )

        self.pos_indices = [i for i, r in enumerate(self.records) if len(r["points"]) > 0]
        self.all_indices = list(range(len(self.records)))
        self.val_tiles = self._build_validation_tiles() if (not training and self.val_sampling.get("enabled", False)) else None

    def __len__(self):
        if self.training:
            return self.samples_per_epoch
        if self.val_tiles is not None:
            return len(self.val_tiles)
        return len(self.records)

    def _sample_record(self):
        positive_fraction = float(self.cfg["data"].get("positive_fraction", 0.4))
        if self.training and random.random() < positive_fraction and self.pos_indices:
            return self.records[random.choice(self.pos_indices)], "positive"
        return self.records[random.choice(self.all_indices)], "mixed"

    def _choose_tile_from_shape_prior(self, w, h, prior, points, mode, rng=random):
        if mode == "positive" and len(points) > 0:
            x, y, _ = rng.choice(points)
            jitter = self.tile // 3
            x0 = int(x - self.tile / 2 + rng.randint(-jitter, jitter))
            y0 = int(y - self.tile / 2 + rng.randint(-jitter, jitter))
            return valid_tile_origin(x0, y0, w, h, self.tile)

        r = rng.random()
        high_prior_frac = float(self.cfg["data"].get("high_prior_negative_fraction", 0.3))
        low_prior_frac = float(self.cfg["data"].get("low_prior_fraction", 0.1))
        threshold = float(self.cfg["data"].get("prior_threshold", 0.35))

        if mode == "high_prior" or (mode == "mixed" and r < high_prior_frac):
            if prior.max() > threshold:
                ys, xs = np.where(prior >= threshold)
                if len(xs) > 0:
                    j = rng.randrange(len(xs))
                    return valid_tile_origin(int(xs[j] - self.tile / 2), int(ys[j] - self.tile / 2), w, h, self.tile)

        if mode == "low_prior" or (mode == "mixed" and r < high_prior_frac + low_prior_frac and prior.size > 0):
            for _ in range(20):
                x0 = rng.randint(0, max(0, w - self.tile))
                y0 = rng.randint(0, max(0, h - self.tile))
                if prior[y0:y0 + self.tile, x0:x0 + self.tile].mean() < threshold * 0.25:
                    return x0, y0

        return rng.randint(0, max(0, w - self.tile)), rng.randint(0, max(0, h - self.tile))

    def _build_validation_tiles(self):
        samples_per_image = int(self.val_sampling.get("samples_per_image", 8))
        seed = int(self.val_sampling.get("seed", self.cfg.get("seed", 1337)))
        positive_fraction = float(self.val_sampling.get("positive_fraction", self.cfg["data"].get("positive_fraction", 0.4)))
        rng = random.Random(seed)
        positive_count = round(samples_per_image * positive_fraction)
        modes = ["positive"] * positive_count + ["random"] * max(0, samples_per_image - positive_count)

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
            x0, y0 = self._choose_tile_from_shape_prior(w, h, prior, rec["points"], mode)
        elif fixed_origin is not None:
            x0, y0 = fixed_origin
        elif rec["points"]:
            x, y, _ = rec["points"][0]
            x0, y0 = valid_tile_origin(int(x - self.tile / 2), int(y - self.tile / 2), w, h, self.tile)
        else:
            x0, y0 = valid_tile_origin((w - self.tile) // 2, (h - self.tile) // 2, w, h, self.tile)

        tile_img = img[y0:y0 + self.tile, x0:x0 + self.tile]
        tile_prior = prior[y0:y0 + self.tile, x0:x0 + self.tile]

        if tile_img.shape[0] != self.tile or tile_img.shape[1] != self.tile:
            tile_img = cv2.copyMakeBorder(tile_img, 0, self.tile - tile_img.shape[0], 0, self.tile - tile_img.shape[1], cv2.BORDER_REFLECT)
            tile_prior = cv2.copyMakeBorder(tile_prior, 0, self.tile - tile_prior.shape[0], 0, self.tile - tile_prior.shape[1], cv2.BORDER_REFLECT)

        local_pts = points_in_tile(rec["points"], x0, y0, self.tile)
        center_target = make_center_heatmap(self.tile, self.tile, local_pts, self.sigma)
        log_diameter_target, diameter_mask = make_log_diameter_targets(
            self.tile,
            self.tile,
            local_pts,
            sigma=self.sigma,
            mask_threshold=self.mask_threshold,
        )

        return {
            "image": torch.from_numpy(tile_img).float().permute(2, 0, 1) / 255.0,
            "prior": torch.from_numpy(tile_prior[None]).float(),
            "center_target": torch.from_numpy(center_target[None]).float(),
            "log_diameter_target": torch.from_numpy(log_diameter_target[None]).float(),
            "diameter_mask": torch.from_numpy(diameter_mask[None]).float(),
            "origin": torch.tensor([x0, y0]),
            "num_points": len(local_pts),
        }


class MultiScaleDiameterLesionTileDataset(Dataset):
    """
    Multiscale diameter dataset.

    CSV columns:
      image_path_1: full-resolution RGB image
      image_path_2: 2x downsampled RGB image
      image_path_4: 4x downsampled RGB image
      image_path_8: 8x downsampled RGB image
      points_json: JSON list [[x,y,diameter_px], ...] in full-image coordinates
    """

    def __init__(self, csv_path: str, cfg: Dict, training: bool = True):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.training = training
        self.sigma = float(cfg["data"].get("sigma_px", 6))
        self.mask_threshold = float(cfg["data"].get("diameter_mask_threshold", 0.25))
        self.samples_per_epoch = int(cfg["data"].get("train_samples_per_epoch", len(self.df)))
        self.tile_sizes = list(cfg["data"].get("multiscale_tile_sizes", [768, 624, 432, 224]))
        self.scales = list(cfg["data"].get("multiscale_scales", [1, 2, 4, 8]))

        column_names = self.df.columns.to_list()
        points_column = "points_json" if "points_json" in column_names else "pts_json"

        self.records = []
        for _, row in self.df.iterrows():
            self.records.append(
                {
                    "image_path_1": row["image_path_1"],
                    "image_path_2": row["image_path_2"],
                    "image_path_4": row["image_path_4"],
                    "image_path_8": row["image_path_8"],
                    "points": parse_points(row.get(points_column, "[]")),
                }
            )

        self.pos_indices = [i for i, r in enumerate(self.records) if len(r["points"]) > 0]
        self.all_indices = list(range(len(self.records)))

    def __len__(self):
        if self.training:
            return self.samples_per_epoch * len(self.scales)
        return len(self.records)

    def _sample_record(self):
        positive_fraction = float(self.cfg["data"].get("positive_fraction", 0.4))
        if self.training and random.random() < positive_fraction and self.pos_indices:
            return self.records[random.choice(self.pos_indices)], "positive"
        return self.records[random.choice(self.all_indices)], "mixed"

    def _choose_tile(self, img, points, mode, rng=random):
        h, w = img.shape[:2]
        tile = self.tile_sizes[0]
        if mode == "positive" and len(points) > 0:
            x, y, _ = rng.choice(points)
            jitter = tile // 3
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
                local.append(((x - x0) / scale, (y - y0) / scale, d / scale))
        return local

    def __getitem__(self, idx):
        rec, mode = self._sample_record() if self.training else (self.records[idx], "positive")

        imgs = [
            read_rgb(rec["image_path_1"]),
            read_rgb(rec["image_path_2"]),
            read_rgb(rec["image_path_4"]),
            read_rgb(rec["image_path_8"]),
        ]

        x0, y0 = self._choose_tile(imgs[0], rec["points"], mode)
        tiles = [
            img[
                y0 // scale:y0 // scale + tile_size,
                x0 // scale:x0 // scale + tile_size,
            ]
            for img, tile_size, scale in zip(imgs, self.tile_sizes, self.scales)
        ]

        for i, tile in enumerate(tiles):
            expected = self.tile_sizes[i]
            h, w = tile.shape[:2]
            if h != expected or w != expected:
                border_type = cv2.BORDER_REFLECT if h > 1 and w > 1 else cv2.BORDER_REPLICATE
                tiles[i] = cv2.copyMakeBorder(tile, 0, expected - h, 0, expected - w, border_type)

        local_pts_by_scale = [
            self._points_in_scaled_tile(rec["points"], x0, y0, tile_size, scale)
            for tile_size, scale in zip(self.tile_sizes, self.scales)
        ]

        center_targets = [
            make_center_heatmap(tile_size, tile_size, pts, max(1.0, self.sigma / scale))
            for tile_size, scale, pts in zip(self.tile_sizes, self.scales, local_pts_by_scale)
        ]
        diameter_targets_and_masks = [
            make_log_diameter_targets(
                tile_size,
                tile_size,
                pts,
                sigma=max(1.0, self.sigma / scale),
                mask_threshold=self.mask_threshold,
            )
            for tile_size, scale, pts in zip(self.tile_sizes, self.scales, local_pts_by_scale)
        ]
        log_diameter_targets = [target for target, _ in diameter_targets_and_masks]
        diameter_masks = [mask for _, mask in diameter_targets_and_masks]

        return {
            "images": [torch.from_numpy(tile).float().permute(2, 0, 1) / 255.0 for tile in tiles],
            "center_targets": [torch.from_numpy(target[None]).float() for target in center_targets],
            "log_diameter_targets": [torch.from_numpy(target[None]).float() for target in log_diameter_targets],
            "diameter_masks": [torch.from_numpy(mask[None]).float() for mask in diameter_masks],
            "origins": [torch.tensor([x0 // scale, y0 // scale]) for scale in self.scales],
            "num_points": len(local_pts_by_scale[0]),
        }
