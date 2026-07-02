import argparse
import json
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import integrate

import argparse
import os


def parse_gt(s):

    if s is None or str(s).strip() == "" or str(s).lower() == "nan":
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(json.loads(s), dtype=np.float32)
    #arr = arr[:, :2]  # ignore diameter if present
    return arr


def match_points(pred, gts, tol):
    #print(f"match_points: pred.shape={pred.shape}, gt.shape={gt.shape}, tol={tol}")
    if len(gts) == 0:
        return 0, len(pred), 0, np.zeros((0, 2), dtype=np.float32), None, None

    gt = gts[:, :2] 
    tol = gts[:, -1]  

    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    i = 0
    tp_points = []
    unmatched_points = []
    score = 0
    for x,y in pred:
        d = np.sqrt(((gt - np.array([x, y])) ** 2).sum(axis=1))
        j = int(np.argmin(d)) if len(d) else -1
    
        i += 1
        if j >= 0 and d[j] <= tol[j] / 2 and not used[j]:
            tp_points.append([x, y, score])
            used[j] = True
            tp += 1
        else:
            unmatched_points.append([x, y, score])
    fp = len(pred) - tp
    fn = len(gt) - tp
    fn_points = gt[~used]
    fn_indices = np.where(~used)[0]
    return tp, fp, fn, fn_points, fn_indices, {"tp_points": tp_points, "unmatched_points": unmatched_points}

def test(manifest, threshold=0.3):
    df = pd.read_csv(manifest)
    total_tp = total_fp = total_fn = 0
    num_candidates = 0
    fn_rows = []
    for i, row in df.iterrows():
        gt = parse_gt(row["points_json"])
        p = pd.read_csv(row["pred_csv"])
        p = p[p["score"] >= threshold]
        num_candidates += len(p)
        pred = p[["x", "y"]].to_numpy(dtype=np.float32)
        out = match_points(pred, gt, tol=24)
        
        tp, fp, fn, fn_points, fn_indices, extra = out
  # Create dummy indices if not returned
        total_tp += tp; total_fp += fp; total_fn += fn
        fn_rows.append({
                "image_path": row["image_path"],
                "points": json.dumps(fn_points.tolist(), separators=(",", ":")),
            })

    sens = total_tp / max(1, total_tp + total_fn)
    prec = total_tp / max(1, total_tp + total_fp)
    fpi = total_fp / max(1, len(df))

    return [ threshold, total_tp, sens, num_candidates, num_candidates - total_tp, np.round((num_candidates - total_tp ) / len(df)) ], pd.DataFrame(fn_rows, columns=["image_path", "points"]), extra

    print({"sensitivity": sens, "precision": prec, "false_positives_per_image": fpi,
           "tp": total_tp, "fp": total_fp, "fn": total_fn})
    print(fn_points)
    print(fn_indices)

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
        tp, fp, fn, fn_points, _, fp_points = match_points(pred, gt, args.tol)
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
    print([[int(fp[0]), int(fp[1])] for fp in fp_points['unmatched_points']])
    if args.fn_out_csv:
        pd.DataFrame(fn_rows, columns=["image_path", "pred_csv", "x", "y"]).to_csv(args.fn_out_csv, index=False)


if __name__ == "__main__":

    main()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--val_out_dir")
    # args = parser.parse_args()

    # eval_file = args.val_out_dir + "/eval.csv"

    # tp_scores = []
    # unmatched_scores = []

    # fig,ax = plt.subplots(1, 2, figsize=(10, 5))

    # assert os.path.exists(eval_file), "Manifest File Does Not Exist"

    # thresholds = [i / 20 for i in range(1, 20)]
    # df = pd.DataFrame(columns=["Threshold", "Matched Points", "Sensitivity", "Candidates", "Unmatched Points", "Unmatched Points per Image"])
    # for score_threshold in thresholds:
    #     out, fns, extra = test(eval_file, threshold=score_threshold)
    #     df.loc[len(df)] =  out
    # #print(df)

    
    # filtered = df[df["Sensitivity"] > 0.95]
    # if not len(filtered):
    #     filtered = df[df["Sensitivity"] < 0.95]
    #     result_row = filtered.loc[filtered["Sensitivity"].idxmax()]
    # result_row = filtered.loc[filtered["Threshold"].idxmax()]
    # sens = result_row["Sensitivity"]
    # unmatched_per_image = result_row["Unmatched Points per Image"]
    # q = 1 / (1 + .0111 * unmatched_per_image)
    # mss = 2 * sens * q / (sens + q)
    # print("---" + args.val_out_dir + "---")
    # print("Best Results at Sensitivity > 0.95:")
    # print(f"MSS: {mss}")
    # print(result_row.to_dict())

    # unmatched_points = df["Unmatched Points"].tolist()
    # candidates = df["Candidates"].tolist()
    # sensitivity = df["Sensitivity"].tolist()

    # x = np.array(df["Unmatched Points"], dtype=float)
    # y = np.array(df["Sensitivity"], dtype=float)
    # order = np.argsort(x)
    # x_sorted = x[order]
    # # x_min = np.min(x_sorted)
    # # x_max = np.max(x_sorted)
    # # x_sorted = [(x-min(x_sorted)) / (max(x_sorted) - min(x_sorted)) for x in x_sorted]
    # y_sorted = y[order]

    # auc = integrate.trapezoid(y_sorted, x_sorted)
    # print("AUC", ": ", auc / unmatched_points[0])

    # plt.figure(figsize=(10, 5))
    # plt.plot(unmatched_points, sensitivity, marker='o', label='Sensitivity', color='blue')

    # for x,y in zip(unmatched_points, sensitivity):
    #     plt.annotate(f"{thresholds[unmatched_points.index(x)]}", (x, y), fontsize=8, ha='right', va='bottom')

    # plt.axhline(y=0.95, color='red', linestyle='--', label='Sensitivity = 0.95')
    # plt.axhline(y=0.92, color='green', linestyle='--', label='Sensitivity = 0.92')
    # plt.axhline(y=0.90, color='orange', linestyle='--', label='Sensitivity = 0.90')
    # plt.xlabel('Unmatched Points')
    # plt.ylabel('Sensitivity')
    # plt.title('Sensitivity vs Unmatched Points - AUC: {:.4f}'.format(auc / max(unmatched_points)))
    # plt.legend()
    # plt.savefig(args.val_out_dir + '/FROC.png')

    

