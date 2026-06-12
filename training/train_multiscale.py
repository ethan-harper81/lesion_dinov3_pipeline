import argparse
import os
import subprocess
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from lesion_dinov3.dataset import LesionTileDataset, MultiScaleLesionTileDataset
from lesion_dinov3.losses import focal_bce_with_logits, weighted_bce_with_logits, tversky_loss
from lesion_dinov3.model import DINOv3LesionDetector
from lesion_dinov3.utils import seed_everything, format_duration

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import timeit

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = cfg if cfg != None else load_cfg(args.config)
    seed_everything(int(cfg.get("seed", 1337)))
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    nw = cfg["data"].get("num_workers", 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MultiScaleLesionTileDataset(cfg["data"]["train_csv"], cfg, training=True)
    print(len(train_ds), "training samples")
    val_ds = LesionTileDataset(cfg["data"]["val_csv"], cfg, training=False)
    print(len(val_ds), "validation samples")

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4), 
        pin_memory=True, 
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4), 
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None )
    
    model = DINOv3LesionDetector(
        cfg["model"]["hf_model_name"],
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
        use_prior=cfg["model"].get("use_prior", True),
        head_channels=cfg["model"].get("head_channels", 256)
    ).to(device)
    print(model.use_prior)
    print(model.head)

    def compute_loss(logits, target, mode = "train"):

        if cfg["train"].get("loss", "focal_bce") == "weighted_bce":
            loss = weighted_bce_with_logits(logits, target, cfg["train"].get("pos_weight", 20.0))
        else:
            loss = focal_bce_with_logits(logits, target, cfg["train"].get("focal_gamma", 2.0), cfg["train"].get("pos_weight", 20.0))
        return loss
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5 ))
    
    weights = [0.1, 0.25, 0.5, 1.0]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["head_lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True)) and device == "cuda")
    
    training_loss_history = []
    validation_loss_history =[]

    t0 = timeit.default_timer()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for batch in pbar:
            images = [x.to(device, non_blocking=True) for x in batch["images"]]
            targets = [y.to(device, non_blocking=True) for y in batch["targets"]]

            opt.zero_grad(set_to_none = True)
            
            prior = None # prior for 8x None
            losses = []
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                for i in reversed(range(len(images))):
                    if i == 0: #prior for 1x None
                        prior = None
                    
                    logits = model(images[i], prior)
                    loss = compute_loss(logits, targets[i])
                    losses.append(loss)
                    prior = torch.sigmoid(logits).detach()
                    prior = F.interpolate(prior, size=images[i-1].shape[-2:], mode ="bilinear", align_corners=False)
            
            loss = sum(weight * loss_i for weight, loss_i in zip(weights, losses))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            pbar.set_postfix(loss=running / max(1, pbar.n))
        running /= max(1, len(train_loader))
        training_loss_history.append(running)

        model.eval()
        val_loss = 0.0
        # with torch.no_grad():
        #     for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
        #         image = batch["image"].to(device, non_blocking=True)
        #         prior = batch["prior"].to(device, non_blocking=True)
        #         target = batch["target"].to(device, non_blocking=True)
        #         logits = model(image, prior)
        #         val_loss += float(compute_loss(logits, target))
        # val_loss /= max(1, len(val_loader))
        # print(f"epoch={epoch} val_loss={val_loss:.6f}")
        # validation_loss_history.append(val_loss)

        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}
        if epoch % int(cfg["train"].get("save_every", 1)) == 0:
            torch.save(ckpt, os.path.join(cfg["train"]["output_dir"], f"epoch_{epoch}.pt"))

    t1 = timeit.default_timer()

    ax[0].plot(list(range(1, len(training_loss_history) + 1)), training_loss_history)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    # ax[1].plot(list(range(1, len(validation_loss_history) + 1)), validation_loss_history)
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Loss')
    # ax[1].set_title('Validation Loss')

    fig.tight_layout()
    plt.savefig(os.path.join(cfg["train"]["output_dir"], "loss.png"))

    try:
        print(f"Total Training time: {format_duration(t1 - t0)}")
    except Exception as e:
        print(f"Error formatting duration: {e}. Total Training time in seconds: {t1 - t0:.2f}")


if __name__ == "__main__":
    main()    