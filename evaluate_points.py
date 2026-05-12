import argparse
import json
import math

import numpy as np
import pandas as pd


def parse_gt(s):
    if s is None or str(s).strip() == "" or str(s).lower() == "nan":
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(json.loads(s), dtype=np.float32)


def match_points(pred, gt, tol):
    if len(gt) == 0:
        return 0, len(pred), 0
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for p in pred:
        d = np.sqrt(((gt - p) ** 2).sum(axis=1))
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with image_path, points_json, pred_csv")
    ap.add_argument("--tol", type=float, default=24)
    ap.add_argument("--score_threshold", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    total_tp = total_fp = total_fn = 0
    for _, row in df.iterrows():
        gt = parse_gt(row["points_json"])
        p = pd.read_csv(row["pred_csv"])
        p = p[p["score"] >= args.score_threshold]
        pred = p[["x", "y"]].to_numpy(dtype=np.float32)
        tp, fp, fn = match_points(pred, gt, args.tol)
        total_tp += tp; total_fp += fp; total_fn += fn

    sens = total_tp / max(1, total_tp + total_fn)
    prec = total_tp / max(1, total_tp + total_fp)
    fpi = total_fp / max(1, len(df))
    print({"sensitivity": sens, "precision": prec, "false_positives_per_image": fpi,
           "tp": total_tp, "fp": total_fp, "fn": total_fn})


if __name__ == "__main__":
    main()
