import argparse
import json
import platform
import sys
from types import MethodType
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lesion_dinov3.model import DINOv3LesionDetector


class ProbabilityTileModel(torch.nn.Module):
    """Fixed-size export wrapper that returns probabilities for one tile."""

    def __init__(self, model: torch.nn.Module, tile_size: int):
        super().__init__()
        self.model = model
        self.tile_size = tile_size
        self.patch_size = int(getattr(model.backbone.config, "patch_size", 16))
        self.hidden_size = int(getattr(model.backbone.config, "hidden_size"))
        self.grid_h = tile_size // self.patch_size
        self.grid_w = tile_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w

    def forward(self, image: torch.Tensor):
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (image - mean) / std

        out = self.model.backbone(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, -self.num_patches :, :]
        feat = patch_tokens.reshape(
            1, self.grid_h, self.grid_w, self.hidden_size
        )
        feat = feat.permute(0, 3, 1, 2).contiguous()

        if self.model.use_prior:
            prior = feat.new_zeros((1, 1, self.grid_h, self.grid_w))
            feat = torch.cat([feat, self.model.prior_proj(prior)], dim=1)

        logits_small = self.model.head(feat)
        logits = torch.nn.functional.interpolate(
            logits_small,
            size=(self.tile_size, self.tile_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits)


def patch_dinov3_vit_for_coreml_export(model: torch.nn.Module, tile_size: int):
    """Avoid dynamic token split math that Core ML Tools cannot lower."""

    try:
        from transformers.models.dinov3_vit import modeling_dinov3_vit
    except ImportError:
        return

    cfg = model.backbone.config
    patch_size = int(getattr(cfg, "patch_size", 16))
    hidden_size = int(getattr(cfg, "hidden_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = hidden_size // num_heads
    half_head_dim = head_dim // 2
    num_patches = (tile_size // patch_size) * (tile_size // patch_size)
    num_prefix_tokens = 1 + int(getattr(cfg, "num_register_tokens", 0))

    rope_embeddings = getattr(model.backbone, "rope_embeddings", None)
    if rope_embeddings is not None:
        with torch.no_grad():
            example = torch.zeros(1, 3, tile_size, tile_size)
            cos, sin = rope_embeddings(example)
        rope_embeddings.register_buffer("_coreml_cos", cos.detach(), persistent=False)
        rope_embeddings.register_buffer("_coreml_sin", sin.detach(), persistent=False)

        def static_rope_forward(self, pixel_values):
            dtype = pixel_values.dtype
            return self._coreml_cos.to(dtype=dtype), self._coreml_sin.to(dtype=dtype)

        rope_embeddings.forward = MethodType(static_rope_forward, rope_embeddings)

    def static_rotate_half(x):
        x1 = x[..., :half_head_dim]
        x2 = x[..., half_head_dim:head_dim]
        return torch.cat((-x2, x1), dim=-1)

    def static_apply_rotary_pos_emb(q, k, cos, sin, **kwargs):
        q_prefix = q[:, :, :num_prefix_tokens, :]
        k_prefix = k[:, :, :num_prefix_tokens, :]
        q_patches = q[:, :, num_prefix_tokens : num_prefix_tokens + num_patches, :]
        k_patches = k[:, :, num_prefix_tokens : num_prefix_tokens + num_patches, :]
        cos_static = cos[:num_patches, :]
        sin_static = sin[:num_patches, :]

        q_patches = (q_patches * cos_static) + (
            static_rotate_half(q_patches) * sin_static
        )
        k_patches = (k_patches * cos_static) + (
            static_rotate_half(k_patches) * sin_static
        )
        q = torch.cat((q_prefix, q_patches), dim=-2)
        k = torch.cat((k_prefix, k_patches), dim=-2)
        return q, k

    modeling_dinov3_vit.apply_rotary_pos_emb = static_apply_rotary_pos_emb
    cfg._attn_implementation = "eager"

    for module in model.modules():
        module_cfg = getattr(module, "config", None)
        if module_cfg is not None and hasattr(module_cfg, "_attn_implementation"):
            module_cfg._attn_implementation = "eager"


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]

    model = DINOv3LesionDetector(
        cfg["model"]["hf_model_name"],
        freeze_backbone=True,
        use_prior=cfg["model"].get("use_prior", True),
        head_channels=cfg["model"].get("head_channels", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def export_coreml(
    checkpoint_path: Path,
    output_path: Path,
    tile_size: int | None,
    deployment_target: str,
    compute_precision: str,
    convert_to: str,
):
    if convert_to == "mlprogram" and platform.system() == "Windows":
        raise RuntimeError(
            "Core ML Tools cannot write mlprogram .mlpackage files on this Windows "
            "environment because the BlobWriter extension is unavailable. Run the "
            "same command on macOS for the production iOS15+ .mlpackage, or use "
            "--convert-to neuralnetwork --deployment-target iOS14 --output "
            "exports/LesionTile768.mlmodel for a Windows-side legacy export."
        )

    import coremltools as ct

    device = torch.device("cpu")
    model, cfg = load_checkpoint_model(checkpoint_path, device)
    tile = int(tile_size or cfg["infer"]["tile_size"])
    patch_dinov3_vit_for_coreml_export(model, tile)

    wrapped = ProbabilityTileModel(model, tile).eval()
    example = torch.rand(1, 3, tile, tile, device=device)

    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, example)

    target = getattr(ct.target, deployment_target)
    precision = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
    }[compute_precision]

    convert_kwargs = {
        "convert_to": convert_to,
        "inputs": [
            ct.TensorType(
                name="image",
                shape=example.shape,
                dtype=float,
            )
        ],
        "outputs": [
            ct.TensorType(name="probability"),
        ],
        "minimum_deployment_target": target,
    }
    if convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = precision

    mlmodel = ct.convert(traced, **convert_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    metadata = {
        "checkpoint": str(checkpoint_path),
        "coreml_model": str(output_path),
        "tile_size": tile,
        "input": {
            "name": "image",
            "shape": [1, 3, tile, tile],
            "dtype": "float32",
            "range": "0..1 RGB, same as inference/infer_full_image.py",
        },
        "output": {
            "name": "probability",
            "shape": [1, 1, tile, tile],
            "range": "0..1 lesion probability",
        },
        "deployment_target": deployment_target,
        "compute_precision": compute_precision,
        "convert_to": convert_to,
        "infer": cfg.get("infer", {}),
        "model": cfg.get("model", {}),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"saved Core ML model: {output_path}")
    print(f"saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export a single-scale lesion tile checkpoint to Core ML."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .mlpackage path, for example exports/LesionTile.mlpackage.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Defaults to checkpoint cfg['infer']['tile_size'].",
    )
    parser.add_argument(
        "--deployment-target",
        default="iOS15",
        choices=["iOS14", "iOS15", "iOS16", "iOS17", "iOS18", "iOS26"],
    )
    parser.add_argument(
        "--compute-precision",
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--convert-to",
        default="mlprogram",
        choices=["mlprogram", "neuralnetwork"],
        help=(
            "Use mlprogram for iOS 15+ export. Try neuralnetwork if exporting on "
            "Windows fails with 'BlobWriter not loaded'."
        ),
    )
    args = parser.parse_args()

    export_coreml(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tile_size=args.tile_size,
        deployment_target=args.deployment_target,
        compute_precision=args.compute_precision,
        convert_to=args.convert_to,
    )


if __name__ == "__main__":
    main()
