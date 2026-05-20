from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import matplotlib.pyplot as plt

import sys

class DINOv3LesionDetector(nn.Module):
    def __init__(self, hf_model_name: str, freeze_backbone: bool = True, use_prior: bool = True, head_channels: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_model_name)
        self.use_prior = use_prior

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.backbone.config, "embed_dim")

        prior_ch = 32 if use_prior else 0
        if use_prior:
            self.prior_proj = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(hidden + prior_ch, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels // 2, 1, 1),
        )

    def _tokens_to_grid(self, last_hidden_state: torch.Tensor, image_hw: Tuple[int, int]):
        # [B, prefix+N, C] -> [B, C, Gh, Gw]. DINOv3 may include CLS/register
        # tokens before the patch tokens, so keep only the expected patch suffix.
        b, n, c = last_hidden_state.shape
        #print(f"last_hidden_state shape: {last_hidden_state.shape}")
        patch_size = getattr(self.backbone.config, "patch_size", 16)
        gh, gw = image_hw[0] // patch_size, image_hw[1] // patch_size
        num_patches = gh * gw

        if n >= num_patches:
            patch_tokens = last_hidden_state[:, -num_patches:, :]
        else:
            # Fallback for unusual checkpoints that only expose patch tokens.
            patch_tokens = last_hidden_state[:, 1:, :]
            gh = int(round((patch_tokens.shape[1] * image_hw[0] / image_hw[1]) ** 0.5))
            gw = patch_tokens.shape[1] // gh

        if gh * gw != patch_tokens.shape[1]:
            raise RuntimeError(
                f"Cannot reshape {patch_tokens.shape[1]} patch tokens into a grid for image size {image_hw}."
            )
        return patch_tokens.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, image: torch.Tensor, prior: torch.Tensor | None = None):
        # image expected normalized by HF? Here we use ImageNet-like normalization below.
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (image - mean) / std
        out = self.backbone(pixel_values=x)
        feat = self._tokens_to_grid(out.last_hidden_state, image.shape[-2:])

        if self.use_prior:
            if prior is None:
                prior = torch.zeros((image.shape[0], 1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
            psmall = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = torch.cat([feat, self.prior_proj(psmall)], dim=1)

        logits_small = self.head(feat)
        
        output = F.interpolate(logits_small, size=image.shape[-2:], mode="bilinear", align_corners=False)
        debug = False
        if debug:
            print(output[0, 0].shape)
            print(torch.min(output[0, 0]), torch.max(output[0, 0]))
            print(logits_small[0, 0].shape)
            print(torch.min(logits_small[0, 0]), torch.max(logits_small[0, 0]))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("Input Image")
            axes[1].imshow(logits_small[0, 0].detach().cpu().numpy(), cmap="hot")
            axes[1].set_title("Logits (Small)")
            axes[2].imshow(output[0, 0].detach().cpu().numpy(), cmap="hot")
            axes[2].set_title("Output (Upsampled)")
            plt.tight_layout()
            plt.savefig("debug_output.png")
            plt.close()
            sys.exit(0)
        
        return output