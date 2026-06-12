# Core ML iPhone Single-Scale Inference

This repo's Python inference has two parts:

1. A tile model: `DINOv3LesionDetector` maps one RGB tile to lesion logits.
2. Full-image logic: slide tiles over the image, run the model, apply sigmoid, blend overlaps with a Hann window, then run threshold plus NMS.

`tools/export_coreml_tile_model.py` exports part 1 for iPhone. It wraps the checkpoint model so the Core ML output is already a probability tile.

## Export from a checkpoint

Install Core ML Tools in the Python environment you use for export:

```powershell
pip install coremltools
```

Export a specific checkpoint:

```powershell
python tools/export_coreml_tile_model.py `
  --checkpoint runs/lesion_dinov3_vits_original/best.pt `
  --output exports/LesionTile.mlpackage
```

Use that default `mlprogram` export on macOS for the final iPhone model. On Windows, Core ML Tools may convert the graph but fail while writing the `.mlpackage` with `BlobWriter not loaded`. If you need a Windows-side smoke test or legacy `.mlmodel`, use the neural network backend:

```powershell
python tools/export_coreml_tile_model.py `
  --checkpoint runs/lesion_dinov3_vits_multiscale_no_final_prior/epoch_10.pt `
  --output exports/LesionTile224.mlmodel `
  --tile-size 224 `
  --convert-to neuralnetwork `
  --deployment-target iOS14 `
  --compute-precision float32
```

For production on a current iPhone, prefer the default `--convert-to mlprogram` and run it on a Mac:

```powershell
python tools/export_coreml_tile_model.py `
  --checkpoint runs/lesion_dinov3_vits_multiscale_no_final_prior/epoch_10.pt `
  --output exports/LesionTile768.mlpackage `
  --tile-size 768 `
  --convert-to mlprogram `
  --deployment-target iOS15 `
  --compute-precision float16
```

For a different checkpoint, replace `--checkpoint`. The tile size defaults to the checkpoint's `cfg["infer"]["tile_size"]`, currently 768 in `configs/config.yaml`. Override it only if the checkpoint supports the requested size:

```powershell
python tools/export_coreml_tile_model.py `
  --checkpoint runs/lesion_dinov3_vits_multiscale_no_final_prior/best.pt `
  --output exports/LesionTile768.mlpackage `
  --tile-size 768
```

The exported model contract is:

- input `image`: `Float32` `MLMultiArray`, shape `[1, 3, tile, tile]`, RGB, values `0..1`
- output `probability`: `Float32`/`Float16` `MLMultiArray`, shape `[1, 1, tile, tile]`, values `0..1`

Add the `.mlpackage` to the Xcode project and let Xcode compile it into an `.mlmodelc` bundle.

## iPhone inference loop

For single-scale full-image inference on iPhone, mirror `inference/infer_full_image.py`:

1. Decode the image as RGB.
2. Use `tile_size` and `stride` from the checkpoint metadata JSON or config. The current single-scale defaults are `tile_size=768`, `stride=512`.
3. Generate tile origins so the right and bottom edges are covered.
4. For each tile, reflect-pad edge tiles to `tile_size x tile_size`.
5. Convert pixels to a `[1, 3, tile, tile]` `MLMultiArray` in RGB channel order with values divided by 255.
6. Run the Core ML model.
7. Crop padded output back to the real tile area.
8. Add `probability * hannWindow` into a full-size score buffer and add `hannWindow` into a weight buffer.
9. Divide `score / max(weight, 1e-6)` to get the heatmap.
10. Extract candidates using threshold and NMS. Current defaults are `threshold=0.30`, `nms_radius_px=30`, `max_candidates=5000`.

The checkpoint may have `model.use_prior: true`, but this exporter calls the model with `prior=None`, which matches a zero prior for single-scale detector-only inference.
