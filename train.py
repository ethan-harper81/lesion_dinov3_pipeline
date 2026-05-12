import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesion_dinov3.dataset import LesionTileDataset
from lesion_dinov3.losses import focal_bce_with_logits, weighted_bce_with_logits
from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.utils import seed_everything


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = LesionTileDataset(cfg["data"]["train_csv"], cfg, training=True)
    val_ds = LesionTileDataset(cfg["data"]["val_csv"], cfg, training=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["data"].get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"].get("num_workers", 4), pin_memory=True)

    model = DINOv3LesionDetector(
        cfg["model"]["hf_model_name"],
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
        use_prior=cfg["model"].get("use_prior", True),
        head_channels=cfg["model"].get("head_channels", 256),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device == "cuda")

    def compute_loss(logits, target):
        if cfg["train"].get("loss", "focal_bce") == "weighted_bce":
            return weighted_bce_with_logits(logits, target, cfg["train"].get("pos_weight", 20.0))
        return focal_bce_with_logits(logits, target, cfg["train"].get("focal_gamma", 2.0), cfg["train"].get("pos_weight", 20.0))

    best_val = float("inf")
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["prior"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(image, prior)
                loss = compute_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            pbar.set_postfix(loss=running / max(1, pbar.n))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                image = batch["image"].to(device, non_blocking=True)
                prior = batch["prior"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                logits = model(image, prior)
                val_loss += float(compute_loss(logits, target))
        val_loss /= max(1, len(val_loader))
        print(f"epoch={epoch} val_loss={val_loss:.6f}")

        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}
        if epoch % int(cfg["train"].get("save_every", 1)) == 0:
            torch.save(ckpt, os.path.join(cfg["train"]["output_dir"], f"epoch_{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg["train"]["output_dir"], "best.pt"))


if __name__ == "__main__":
    main()
