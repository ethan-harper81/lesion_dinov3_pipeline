import json
import numpy as np
# with open(annotation_path, 'r') as f:
#     annotation = json.load(f)
#     points = [[ann['x'] * 8, ann['y'] * 8, ann['diameter_pixel'] * 8] for ann in annotation['annotations']]

from lesion_dinov3.utils import make_diameter_aware_heatmap, make_heatmap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw
import os
import csv
import pandas as pd

out_dir = 'validation_outputs/multiscale_epoch_9'

csvs = os.listdir(out_dir)

csvs = [c for c in csvs if ".csv" in c]
side= ['front', "back", "left", "right"]

val_csv = 'validation_outputs/multiscale_epoch_9/eval.csv'

val_df = pd.read_csv(val_csv)

fn_csv = 'validation_outputs\\multiscale_epoch_9\\false_negative_0_75.csv'

fn_df = pd.read_csv(fn_csv)

for i, c in enumerate(csvs):
    if c == 'eval.csv' or c == 'false_negative_0_75.csv':
        continue
    df = pd.read_csv(os.path.join(out_dir,c))
    if len(df) > 0:
        points = [[row['x'], row['y'], row['score']] for _, row in df.iterrows()]
    else:
        points = []
    
    save_name = c.split(".jpg")[0]
    prefix = c.split("_candidates")[0]
    prefix = prefix.split('_')
    split = prefix.index('wide') - 1
    patient_name = "_".join(prefix[:split])
    image_name = patient_name + "\\" + prefix[split] + "\\" + "_".join(prefix[split + 1:])
    validataion_row = "C:/Users/davoo/Development/LumoLesionClassifier/" + image_name
    row = val_df.loc[val_df['image_path'] == validataion_row]
    og_points = json.loads(row['points_json'].iloc[0])
    row = fn_df.loc[fn_df['image_path'] == validataion_row]
    fn_points = json.loads(row["points"].iloc[0])
    print(image_name)
    path = os.path.join("C:\\Users\\davoo\\Development\\LumoLesionClassifier", image_name)
    
    threshold = 0.75

    img = Image.open(path)
    draw = ImageDraw.Draw(img)

    values = [v for _,_,v in points]

    vmax = max(values)
    vmin = min(values)

    # norm = mcolors.Normalize(vmin, vmax)
    # cmap = cm.get_cmap("RdYlGn_r")

    box_size = 120
    half = box_size // 2

    points = [point for point in points if point[2] >= 0.75]

    for x,y,score in points:
        # rgba = cmap(norm(score))
        # color = tuple(int(c * 255) for c in rgba[:3])
        if threshold < .75:
            continue
        draw.rectangle(
            [
                (x - half, y - half),
                (x + half, y + half)
            ],
            outline='red',
            width=7
        )
    for x,y,score in og_points:
        draw.circle(
            xy=[x,y],
            radius=5,
            outline='green',
            width=5
        )
    i = 0
    for x,y in fn_points:
    
        # crop = img.crop([
        #     x - 60, y - 60, x + 60, y + 60
        # ])
        # crop.save(f"fn_crops/{save_name}_crop_{i}.png")
        draw.circle(
            xy=[x,y],
            radius = 60,
            outline = 'blue',
            width = 5
        )
        i += 1
    
    img.save(os.path.join("mutliscale_epoch_9_eval_images", save_name + '.png'))


