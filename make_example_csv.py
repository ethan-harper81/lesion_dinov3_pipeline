import argparse
import json
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", default="data/train.csv")
    args = ap.parse_args()
    rows = []
    for name in sorted(os.listdir(args.images_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            rows.append({"image_path": os.path.join(args.images_dir, name), "points_json": json.dumps([]), "prior_path": ""})
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}; fill points_json with [[x,y], ...] and prior_path if available.")


if __name__ == "__main__":
    main()
