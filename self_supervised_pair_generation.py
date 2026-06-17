import argparse
from inference.infer_full_image import load_model, _infer_probability_map, extract_peaks_torch
import torch
from lesion_dinov3.utils import read_rgb
import os
import json
import pandas as pd
import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--im_folder", required=True, help="Folder Consisting of Wide Field of View images")
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint to use for image inference")
    ap.add_argument("--out_csv", default="data_singlescale/psuedo.csv", help="CSV path to write psuedo annotation data")
    ap.add_argument("--threshold", default=0.9, help="Threshold for Psuedo Positive points")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    device = next(model.parameters()).device

    imgs = []

    for root, dirs, files in os.walk(args.im_folder):
        for file in files:
            if ".jpg" in file:
                imgs.append(os.path.join(args.im_folder, root, file))


    points_json = []

    total_high_prob_lesions = 0

    for path in tqdm.tqdm(imgs, desc="Image Inference"):
        img = read_rgb(path)
        h, w = img.shape[:2]
        tile = int(cfg["infer"]["tile_size"])
        stride = int(cfg["infer"]["stride"])

        full_prob_gpu = _infer_probability_map(path, None, model, cfg, tile, stride)

        full_prob = full_prob_gpu.cpu().numpy()

        points, scores = extract_peaks_torch(
            full_prob_gpu,
            threshold=float(cfg["infer"]["threshold"]),
            radius=int(cfg["infer"]["nms_radius_px"]),
            max_candidates=int(cfg["infer"].get("max_candidates", 5000)),
        )

        #points = points[scores > args.threshold]

        total_high_prob_lesions += len(points)

        complete = [[float(x), float(y), float(score)] for (x,y), score in zip(points, scores)]

        csv_value = json.dumps(complete, separators=(",", ":"))
        points_json.append(csv_value)
    df = pd.DataFrame(
        {
            "image_path": imgs,
            "points_json": points_json,
            "prior_path": [None] * len(imgs),
        }
    )

    print("Total High Probability Lesions: ", total_high_prob_lesions)
    print("High Probability Lesions Per Image: ", total_high_prob_lesions / len(imgs))

    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()