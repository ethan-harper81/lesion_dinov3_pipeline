import argparse
import os
import timeit

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from lesion_dinov3.diameter_dataset import DiameterLesionTileDataset, MultiScaleDiameterLesionTileDataset
from lesion_dinov3.diameter_model import DINOv3LesionDiameterDetector
from lesion_dinov3.losses import focal_bce_with_logits, weighted_bce_with_logits
from lesion_dinov3.utils import format_duration, seed_everything
from training.train_diameter import diameter_loss


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_diameter_multiscale.yaml")
    args = ap.parse_args()
    cfg = cfg if cfg is not None else load_cfg(args.config)

    seed_everything(int(cfg.get("seed", 1337)))
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    nw = int(cfg["data"].get("num_workers", 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MultiScaleDiameterLesionTileDataset(cfg["data"]["train_csv"], cfg, training=True)
    val_ds = DiameterLesionTileDataset(cfg["data"]["val_csv"], cfg, training=False)
    print(len(train_ds), "training samples")
    print(len(val_ds), "validation samples")

    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": nw,
        "pin_memory": True,
        "persistent_workers": nw > 0,
        "prefetch_factor": 2 if nw > 0 else None,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = DINOv3LesionDiameterDetector(
        cfg["model"]["hf_model_name"],
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
        use_prior=cfg["model"].get("use_prior", True),
        head_channels=cfg["model"].get("head_channels", 256),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["head_lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device == "cuda")

    lambda_size = float(cfg["train"].get("lambda_size", 0.25))
    scale_weights = list(cfg["train"].get("multiscale_loss_weights", [0.1, 0.25, 0.5, 1.0]))

    def compute_pair_loss(center_logits, pred_log_diameter, center_target, log_diameter_target, diameter_mask):
        if cfg["train"].get("loss", "focal_bce") == "weighted_bce":
            center_loss = weighted_bce_with_logits(center_logits, center_target, cfg["train"].get("pos_weight", 20.0))
        else:
            center_loss = focal_bce_with_logits(
                center_logits,
                center_target,
                cfg["train"].get("focal_gamma", 2.0),
                cfg["train"].get("pos_weight", 20.0),
            )
        size_loss = diameter_loss(pred_log_diameter, log_diameter_target, diameter_mask)
        return center_loss + lambda_size * size_loss, center_loss.detach(), size_loss.detach()

    def compute_validation_loss(batch):
        image = batch["image"].to(device, non_blocking=True)
        center_target = batch["center_target"].to(device, non_blocking=True)
        log_diameter_target = batch["log_diameter_target"].to(device, non_blocking=True)
        diameter_mask = batch["diameter_mask"].to(device, non_blocking=True)

        center_logits, pred_log_diameter = model(image, None)
        loss, _, _ = compute_pair_loss(
            center_logits,
            pred_log_diameter,
            center_target,
            log_diameter_target,
            diameter_mask,
        )
        return loss

    training_loss_history = []
    validation_loss_history = []
    best_val = float("inf")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    t0 = timeit.default_timer()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            images = [x.to(device, non_blocking=True) for x in batch["images"]]
            center_targets = [y.to(device, non_blocking=True) for y in batch["center_targets"]]
            log_diameter_targets = [y.to(device, non_blocking=True) for y in batch["log_diameter_targets"]]
            diameter_masks = [m.to(device, non_blocking=True) for m in batch["diameter_masks"]]

            opt.zero_grad(set_to_none=True)
            losses = []
            prior = None
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                for i in reversed(range(len(images))):
                    if i == 0:
                        prior = None

                    center_logits, pred_log_diameter = model(images[i], prior)
                    loss, center_loss, size_loss = compute_pair_loss(
                        center_logits,
                        pred_log_diameter,
                        center_targets[i],
                        log_diameter_targets[i],
                        diameter_masks[i],
                    )
                    losses.append(loss)

                    if i > 0:
                        prior = torch.sigmoid(center_logits).detach()
                        prior = F.interpolate(prior, size=images[i - 1].shape[-2:], mode="bilinear", align_corners=False)

            loss = sum(weight * loss_i for weight, loss_i in zip(scale_weights, losses))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            pbar.set_postfix(
                loss=running / max(1, pbar.n),
                center=float(center_loss),
                size=float(size_loss),
            )

        train_loss = running / max(1, len(train_loader))
        training_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                val_loss += float(compute_validation_loss(batch).detach())
        val_loss /= max(1, len(val_loader))
        validation_loss_history.append(val_loss)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ax[0].clear()
        ax[0].plot(list(range(1, len(training_loss_history) + 1)), training_loss_history)
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Training Loss")
        ax[1].clear()
        ax[1].plot(list(range(1, len(validation_loss_history) + 1)), validation_loss_history)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].set_title("Validation Loss")
        fig.tight_layout()
        plt.savefig(os.path.join(cfg["train"]["output_dir"], "loss.png"))

        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}
        if epoch % int(cfg["train"].get("save_every", 1)) == 0:
            torch.save(ckpt, os.path.join(cfg["train"]["output_dir"], f"epoch_{epoch}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg["train"]["output_dir"], "best.pt"))

    t1 = timeit.default_timer()
    try:
        print(f"Total Training time: {format_duration(t1 - t0)}")
    except Exception as e:
        print(f"Error formatting duration: {e}. Total Training time in seconds: {t1 - t0:.2f}")


if __name__ == "__main__":
    main()
