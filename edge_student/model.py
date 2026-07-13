from transformers import AutoModel, AutoTokenizer

import torch
import troch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, hf_model_name: str, freeze_backbone: bool = True, head_channels: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.backbone.config, "embed_dim")

        self.head = nn.Sequential(
            nn.Conv2d(hidden, head_channels, 3, padding=1),
            nn.GroupNorm(8, head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(head_channels, head_channels // 2, 3, padding=1),
            nn.GroupNorm(8, head_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(head_channels // 2, 1, 1),
        )

    def _tokens_to_grid(self, last_hidden_state: torch.Tensor, image_hw: Tuple[int, int]):
        b, n, c = last_hidden_state.shape
        patch_size = getattr(self.backbone.config, "patch_size", 16)
        gh, gw = image_hw[0] // patch_size, image_hw[1] // patch_size
        num_patches = gh * gw

        if n >= num_patches:
            last_hidden_state = last_hidden_state[:, -num_patches:, :]
        
        grid = last_hidden_state.transpose(1, 2).reshape(b, c, gh, gw)
        return grid