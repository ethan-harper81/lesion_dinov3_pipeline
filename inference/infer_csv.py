'''
Infer a set of images from a csv
'''

import argparse
from lesion_dinov3.utils import load_model, format_duration
from inference.infer_full_image import run_inference, run_multiscale_inference
import pandas as pd
import torch
import os
from tqdm import tqdm
import timeit

def _create_eval(val_output_dir, val_csv):

    df = pd.read_csv(val_csv)
    image_and_points = [(row["image_path"], row["points_json"]) for _, row in df.iterrows()]
    image_and_points.sort(key=lambda x: x[0])
    image_paths, points_jsons = zip(*image_and_points)

    val_files = os.listdir(val_output_dir)

    candidate_files = [f for f in val_files if f.endswith("_candidates.csv")]
    candidate_files.sort()

    output_df = pd.DataFrame(columns=["image_path", "points_json", "pred_csv"])

    for i, file in enumerate(candidate_files):
        image_file = image_paths[i]
        points_json = points_jsons[i]
        pred_csv = os.path.join(val_output_dir, file)

        output_df.loc[len(output_df)] = [image_file, points_json, pred_csv]
    
    eval_csv = f"{val_output_dir}/eval.csv"

    output_df.to_csv(eval_csv, index=False)

    return eval_csv

def main():

    duration = 0.0

    t0 = timeit.default_timer()

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="csv Document containing any number of rows image_path, points_json, prior_path")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint model from which to run inference")
    ap.add_argument("--out_dir", required=True, help="Folder you would like to save outputs from inferece")
    ap.add_argument("--multiscale", action="store_true")
    args = ap.parse_args()

    assert os.path.exists(args.manifest), "Manifest CSV does not exist, check path"
    assert os.path.exists(args.checkpoint), "Checkpoint .pt does not extst, check path"

    if os.path.exists(args.out_dir):
        ans = input("The out_dir already exists, continuing may overwrite existing data. Would you like to continue ?(y/n):")
        if ans == "n":
            exit(0)
        

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.manifest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device, diameter=False)

    t1 = timeit.default_timer()

    num_images = 0
    for i, row in tqdm(df.iterrows(), desc="Image Set Inference"):
        if args.multiscale:
            _,_,_, time = run_multiscale_inference(
                row["image_path"],
                model,
                cfg,
                args.out_dir,
                verbose = False
            )
        else:
            _,_,_, time = run_inference(
                row["image_path"],
                prior_path=None, 
                model=model,
                cfg=cfg,
                out_dir=args.out_dir,
                verbose = False
            )
        duration += time
        num_images += 1
    
    avg_inference_duration = duration / num_images
    duration += (t1-t0)

    try:
        print(f"Ran Inference on {num_images} Images \nTotal Time: {format_duration(duration)} \nAvg Inference Time: {avg_inference_duration}")
    except Exception as e:
        print(f"Ran Inference on {num_images} Images \nTotal Time: {duration} \nAvg Inference Time: {avg_inference_duration}")

    print(f"All outputs saved to {args.out_dir}")

    eval_csv = _create_eval(args.out_dir, args.manifest)

    print(f"Created csv for evaluation: {eval_csv}")

if __name__ == "__main__":
    main()
