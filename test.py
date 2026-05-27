np_file = "outputs/1x/wide_00_001_lesion_heatmap.npy"
im = "C:/Users/davoo/Development/LumoLesionClassifier/patient_images/front/wide_00_001.jpg"
og_points = [[5328.88,5875.84,37.36],[5162.08,5547.44,36],[5495.68,5428.72,33.52],[4857.28,5441.76,26.64],[5086.96,4971.04,38.8],[5484.32,4702.96,46.96],[5545.44,4520.4,35.84],[5390.88,4626.96,33.36],[4999.6,4441.76,27.2],[5202.24,4323.04,43.12],[5546.32,4260.16,32.24],[4985.6,4104.4,32],[4980,3877.2,50],[4935.2,3591.6,36.24],[5452,3536.4,35.12],[5385.6,3522.8,48.8],[5202.4,3343.6,40],[5292,3074.8,40],[2662.16,2669.64,39.2],[4962.4,2853.2,43.12],[5152.8,2798,35.76],[5134.4,2267.6,44.56],[4966.4,1707.6,41.6],[4547.2,1827.6,39.6],[4620.8,1989.36,29.12],[4653.12,2012,50.56],[4904.96,1996.64,36.32],[5307.46,1594.46,26.72],[4296.32,1642.24,44.88],[4075.12,1402.48,64.32],[4454.56,1080.4,58.32],[4356,1230.56,39.28],[4275.28,1344.4,39.12],[4607.92,1274.96,39.6],[5014.8,1735.92,37.52],[95.52,4945.26,154.44],[4934.88,1527.6,26.08],[4480.4,863.2,86.4],[4446.56,676.96,48.56],[4188.32,430.08,60],[3509.73,648.45,35.4],[3587.28,947.92,34.48],[1575.06,2914.75,62.92],[3812.16,256.64,62.8],[3582.88,105.36,58.72],[3546.32,326.56,63.52],[3466.64,412.72,44.24],[3004,365.6,45.36],[3994.24,309.44,37.92],[3660.16,820.88,34.88],[2767.76,112.96,49.28],[2734.56,257.04,46.96],[1819.12,850.64,47.52],[1975.6,762.08,39.36],[1571.2,789.68,35.04],[2010.08,213.04,45.28],[1665.68,246.48,44.56],[1625.28,251.36,32],[1793.6,789.68,49.2],[1839.2,53.2,51.6],[1680,390.8,37.76],[1704,481.2,40.8],[2155.2,584.4,37.36],[1404,100.4,36.56],[906.72,889.84,53.6],[808.64,1090.4,57.84],[628.56,1182,47.28],[727.47,532.99,42.8],[293.2,485.52,45.84],[1146.08,775.52,29.36],[656.56,972.88,45.36],[946.64,1012.72,54.88],[1056.56,954.56,43.68],[1140.64,221.36,66.64],[147.68,48.88,47.84],[614.7,516.43,66.68],[164.96,268.8,50.32],[1964.48,4255.04,67.2],[2519.52,4706,53.84],[2464.48,4706,72.24],[1781.52,5624.96,58.16],[1590.16,5128.64,48.24],[1363.28,4289.44,92.96],[1025.76,4952.88,65.84],[2746.91,3643.94,27.28],[1137.28,5677.44,64.88],[1040.8,3202.8,47.04],[359.84,4309.76,51.36],[200.64,4317.04,53.04],[160.08,4732.08,61.44],[483.84,5008.4,50.96],[379.28,5019.84,100.96]]
og_points = [[point[0] , point[1] , point[2] ] for point in og_points]
csv_path = 'output/wide_00_001_candidates.csv'
annotation_path = 'C:/Users/davoo/Development/LumoServer/data/original_annotation_data/front/wide_00_001.json'

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

out_dir = 'validation_outputs/20_epochs'

csvs = os.listdir(out_dir)

csvs = [c for c in csvs if ".csv" in c]
side= ['front', "back", "left", "right"]

for i, c in enumerate(csvs):
    if c == 'eval.csv':
        continue
    df = pd.read_csv(os.path.join(out_dir,c))
    if len(df) > 0:
        points = [[row['x'], row['y'], row['score']] for _, row in df.iterrows()]
    else:
        points = []
    
    prefix = c.split("_candidates")[0]
    prefix = prefix.split('_')
    split = prefix.index('wide') - 1
    patient_name = "_".join(prefix[:split])
    image_name = patient_name + "\\" + prefix[split] + "\\" + "_".join(prefix[split + 1:])
    path = os.path.join("C:\\Users\\davoo\\Development\\LumoLesionClassifier", image_name)
    
    threshold = 0.75

    img = Image.open(path)
    draw = ImageDraw.Draw(img)

    values = [v for _,_,v in points]

    vmax = max(values)
    vmin = min(values)

    norm = mcolors.Normalize(vmin, vmax)
    cmap = cm.get_cmap("RdYlGn_r")

    box_size = 120
    half = box_size // 2

    for x,y,score in points:
        rgba = cmap(norm(score))
        color = tuple(int(c * 255) for c in rgba[:3])
        draw.rectangle(
            [
                (x - half, y - half),
                (x + half, y + half)
            ],
            outline='red',
            width=10
        )
    save_name = c.split(".jpg")[0]
    img.save(os.path.join("eval_images", save_name + '.png'))


