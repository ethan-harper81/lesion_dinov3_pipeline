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


eval_csv = f'validation_outputs/psuedo_training/epoch_5/eval.csv'

df = pd.DataFrame(columns=["Threshold", "Matched Points", "Sensitivity", "Soft Precision", "F1 Score", "Candidates", "Unmatched Points", "Unmatched Points per Image"])

for threshold in thresholds:
    out= test(eval_csv, threshold=threshold)
    threshold, total_tp, sens, num_candidates, unmatched, upi = out[0]
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

sensitivity_threshold = 0.95

filtered = df[df["Sensitivity"] > sensitivity_threshold]
if not len(filtered):
    filtered = df[df["Sensitivity"] < sensitivity_threshold]
    result_row = filtered.loc[filtered["Sensitivity"].idxmax()]
result_row = filtered.loc[filtered["Threshold"].idxmax()]
sens = result_row["Sensitivity"]
unmatched_per_image = result_row["Unmatched Points per Image"]
q = 1 / (1 + .0111 * unmatched_per_image)
mss = 2 * sens * q / (sens + q)
print(f"Best Results at Sensitivity > {sensitivity_threshold}:")
print(f"MSS: {mss}")
print(result_row.to_dict())

# plt.plot(thresholds, sensitivities, label='Sensitivity', marker='o')
# plt.plot(thresholds, soft_precisions, label='Soft Precision', marker='o')
# plt.plot(thresholds, F1_scores, label='F1 Score', marker='o')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title(f'ROC')

plt.show()