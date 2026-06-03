import pandas
import csv
import os
from evaluate_points import test
import pandas as pd
import matplotlib.pyplot as plt

thresholds = [i / 20 for i in range(1, 20)]

gamma = 0.2
beta = 1

# fig, ax = plt.subplots(1, 10, figsize=(10, 5))


eval_csv = f'validation_outputs/5_epochs_more_samples/eval.csv'

df = pd.DataFrame(columns=["Threshold", "Matched Points", "Sensitivity", "Soft Precision", "F1 Score", "Candidates", "Unmatched Points", "Unmatched Points per Image"])

for threshold in thresholds:
    out, _ = test(eval_csv, threshold=threshold)
    threshold, total_tp, sens, num_candidates, unmatched, upi = out
    soft_precision = total_tp / max(1, total_tp + unmatched * gamma)
    f1 = (1 + beta**2) * (soft_precision * sens) / max(1e-6, beta**2 * soft_precision + sens)
    df.loc[len(df)] = [
        threshold, total_tp, sens, soft_precision, f1, num_candidates, unmatched, upi
    ]

print(df)

thresholds = df["Threshold"].tolist()
soft_precisions = df["Soft Precision"].tolist()
sensitivities = df["Sensitivity"].tolist()
F1_scores = df["F1 Score"].tolist()

plt.plot(thresholds, sensitivities, label='Sensitivity', marker='o')
plt.plot(thresholds, soft_precisions, label='Soft Precision', marker='o')
plt.plot(thresholds, F1_scores, label='F1 Score', marker='o')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'ROC')

plt.show()