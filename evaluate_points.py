import argparse
import json
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_gt(s):

    if s is None or str(s).strip() == "" or str(s).lower() == "nan":
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(json.loads(s), dtype=np.float32)
    arr = arr[:, :2]  # ignore diameter if present
    return arr


def match_points(pred, gt, tol):
    if len(gt) == 0:
        return 0, len(pred), 0, np.zeros((0, 2), dtype=np.float32)
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    print(f"matching {len(pred)} predicted points to {len(gt)} ground truth points with tol={tol}")
    i = 0
    for p in pred:
        d= np.sqrt(((gt - p) ** 2).sum(axis=1))
        j = int(np.argmin(d)) if len(d) else -1
        
        print(f"pred point {p}  closest gt point {gt[j] if j >= 0 else None} at distance {d[j] if j >= 0 else None}")
        i += 1
        if j >= 0 and d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    fn_points = gt[~used]
    return tp, fp, fn, fn_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with image_path, points_json, pred_csv")
    ap.add_argument("--tol", type=float, default=24)
    ap.add_argument("--score_threshold", type=float, default=0.3)
    ap.add_argument("--fn_out_csv", default=None, help="Optional CSV path to write false negative points")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    total_tp = total_fp = total_fn = 0
    fn_rows = []
    for _, row in df.iterrows():
        gt = parse_gt(row["points_json"])
        p = pd.read_csv(row["pred_csv"])
        p = p[p["score"] >= args.score_threshold]
        pred = p[["x", "y"]].to_numpy(dtype=np.float32)
        tp, fp, fn, fn_points = match_points(pred, gt, args.tol)
        for x, y in fn_points:
            fn_rows.append({
                "image_path": row["image_path"],
                "pred_csv": row["pred_csv"],
                "x": float(x),
                "y": float(y),
            })
        total_tp += tp; total_fp += fp; total_fn += fn

    sens = total_tp / max(1, total_tp + total_fn)
    prec = total_tp / max(1, total_tp + total_fp)
    fpi = total_fp / max(1, len(df))
    print({"sensitivity": sens, "precision": prec, "false_positives_per_image": fpi,
           "tp": total_tp, "fp": total_fp, "fn": total_fn})
    
    fns = [(fn['x'], fn['y']) for fn in fn_rows]
    print(fns)
    if args.fn_out_csv:
        pd.DataFrame(fn_rows, columns=["image_path", "pred_csv", "x", "y"]).to_csv(args.fn_out_csv, index=False)


if __name__ == "__main__":
    main()
