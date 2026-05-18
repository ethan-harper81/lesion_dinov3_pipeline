# DINOv3 Point-Supervised Lesion Heatmap Pipeline

This starter repo trains a DINOv3-backed lesion detector from full-image point annotations and optional full-image lesion prior heatmaps.

## Data format

Create CSVs for train/val:

```csv
image_path,points_json,prior_path
/path/image_001.png,"[[1234,567],[2222,3333]]",/path/image_001_prior.png
/path/image_002.png,"[]",/path/image_002_prior.png
```

Coordinates are full-resolution image coordinates, e.g. in a 9504 x 6336 image.
`prior_path` is optional. If absent, the prior is treated as zeros.

## Install

```bash
pip install -r requirements.txt
```

You may need to replace `model.hf_model_name` in `config.yaml` with the exact DINOv3 checkpoint available in your environment.

## Train

```bash
python train.py --config config.yaml
```

The model samples 768 x 768 tiles from full-resolution images. Positive tiles are centered near annotated lesion points. High-prior negative tiles are sampled from high prior regions that do not contain annotated points.

## Infer on one full-resolution image

```bash
python infer_full_image.py \
  --checkpoint runs/lesion_dinov3/best.pt \
  --image /path/fullres.png \
  --prior /path/prior.png \
  --out_dir outputs
```

Outputs:

- `*_lesion_heatmap.npy`: full-image float lesion probability map
- `*_lesion_heatmap.png`: 8-bit preview
- `*_candidates.csv`: x, y, score candidate points after threshold + NMS

## Evaluate point detections

Prepare a manifest with columns:

```csv
image_path,points_json,pred_csv
/path/image_001.png,"[[1234,567]]",outputs/image_001_candidates.csv
```

Run:

```bash
python evaluate_points.py --manifest val_predictions.csv --tol 24 --score_threshold 0.3
```

To save missed ground-truth points for review:

```bash
python evaluate_points.py --manifest val_predictions.csv --tol 24 --score_threshold 0.3 --fn_out_csv false_negatives.csv
```

## Recommended starting config for 9504 x 6336 images

- tile size: 768
- stride: 512 for inference, 384 if you need more overlap
- NMS radius: approximate lesion diameter in pixels
- target sigma: lesion radius / 2, often 6 to 12 px
- train frozen DINOv3 first; unfreeze later only if the head plateaus

## Notes

The prior is used in two ways:

1. Training sampler: guides where tiles are sampled.
2. Model input: prior crop is fused into the heatmap head after DINOv3 features.

At inference, `prior_blend_alpha` defaults to 0 so the final output is detector-only. Increase it cautiously, e.g. 0.1 or 0.2, only after validation confirms that the prior helps.
