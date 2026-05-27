import pandas as pd
import json
import subprocess
import os
import sys



def infer_validation_files(run):
    validation_csv = "data_final/val.csv"

    df = pd.read_csv(validation_csv)

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}: {row['image_path']}")
        image_path = row["image_path"]
        points_json = row["points_json"]
        prior_path = row["prior_path"]

        checkpoint_path = f"runs/lesion_dinov3_vits_{run}/epoch_19.pt" 
        out_dir = f"validation_outputs/{run}_test"

        subprocess.run([sys.executable, "infer_full_image.py", "--checkpoint", checkpoint_path, "--image", image_path, "--out_dir", out_dir])

def create_eval(run):
    val_output_dir = f"validation_outputs/{run}"
    val_csv = "data_final/val.csv"

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

    output_df.to_csv(f"validation_outputs/{run}_test/eval.csv", index=False)

if __name__ == "__main__":
    #Name of training run to evaluate
    run = "20_epochs"
    infer_validation_files(run)
    create_eval(run)

        
