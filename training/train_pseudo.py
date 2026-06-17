'''
Training Loop Inlcuding Pseudo Positive Labels from Checkpoint
'''

import argparse
import yaml
import torch
from itertools import cycle
from tqdm import tqdm
import os

from torch.utils.data import DataLoader
from lesion_dinov3.dataset import LesionTileDataset
from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.utils import seed_everything, format_duration
from lesion_dinov3.losses import focal_bce_with_logits
from inference.infer_full_image import load_model

import matplotlib.pyplot as plt

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    nw = cfg["data"].get("num_workers", 4)

    manual_ds = LesionTileDataset(cfg["data"]["manual_csv"], cfg, training=True)
    pseudo_ds = LesionTileDataset(cfg["data"]["pseudo_csv"], cfg, training=True)

    batch_size = cfg["train"]["batch_size"]
    assert batch_size % 2 == 0

    half_bs = batch_size // 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    manual_loader = DataLoader(
        manual_ds,
        batch_size=half_bs,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=False,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None ,
        drop_last=True,
    )

    pseudo_loader = DataLoader(
        pseudo_ds,
        batch_size=half_bs,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=False,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None ,
        drop_last=True,
    )

    model, _ = load_model(args.checkpoint, device)
    
    def compute_loss(logits, target, mode = "train"):
        loss = focal_bce_with_logits(logits, target, cfg["train"].get("focal_gamma", 2.0), cfg["train"].get("pos_weight", 20.0))

        return loss
    
    total_loss_history = []
    manual_loss_history = []
    pseudo_loss_history = []
    
    fig, ax = plt.subplots(1,1, figsize=(10,5))

    pseudo_iter = iter(pseudo_loader)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["head_lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device == "cuda")


    for epoch in range(1, int(cfg["train"]["epochs"] + 1)):
        model.train()
        running_total = 0.0
        running_manual = 0.0
        running_pseudo = 0.0
        pbar = tqdm(
            enumerate(manual_loader),
            total=len(manual_loader),
            desc=f"epoch {epoch} train"
        )

        for i , manual_batch in pbar:
            try:
                pseudo_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                pseudo_batch = next(pseudo_iter)

            image = torch.cat([manual_batch["image"], pseudo_batch["image"]], dim = 0).to(device)
            prior = torch.cat([manual_batch["prior"], pseudo_batch["prior"]], dim = 0).to(device)
            manual_target = manual_batch["target"].to(device)
            pseudo_target = pseudo_batch["target"].to(device)

            logits = model(image, prior)

            manual_logits = logits[:half_bs]
            pseudo_logits = logits[half_bs:]

            manual_loss = compute_loss(manual_logits, manual_target)
            pseudo_loss = compute_loss(pseudo_logits, pseudo_target)

            loss = manual_loss + 0.25 * pseudo_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_total += float(loss.detach())
            running_manual += float(manual_loss.detach())
            running_pseudo += float(pseudo_loss.detach())

            pbar.set_postfix(
                loss=f"{running_total/(i+1):.5f}",
                manual=f"{manual_loss.item():.5f}",
                pseudo=f"{pseudo_loss.item():.5f}"
            )

        running_total /= max(1, len(manual_loader))
        running_manual /= max(1, len(manual_loader))
        running_pseudo /= max(1, len(manual_loader))

        print(
            f"epoch={epoch} "
            f"loss={running_total:.5f} "
            f"manual={running_manual:.5f} "
            f"pseudo={running_pseudo:.5f}"
        )

        ax[0].clear()
        ax[0].plot(list(range(1, len(total_loss_history) + 1)), total_loss_history, label = "Total Loss")
        ax[0].plot(list(range(1, len(manual_loss_history) + 1)), manual_loss_history, label = "Manual Loss")
        ax[0].plot(list(range(1, len(pseudo_loss_history) + 1)), pseudo_loss_history, label = "Pseudo Loss")
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[0].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(cfg["train"]["output_dir"], "loss.png"))

        ckpt = {"model": model.state_dict(), "cfg":cfg, "epoch": epoch}

        torch.save(ckpt, os.path.join(cfg["train"]["out_dir"], f"epoch_{epoch}.pt"))

if __name__ == "__main__":
    main()