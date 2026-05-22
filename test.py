np_file = "outputs/1x/wide_00_001_lesion_heatmap.npy"
im = "C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_images_2/front/wide_00_001.jpg"
og_points = [[874.35,2153.35,55.56],[2455.92,6014.72,61.1],[1439.61,1199.05,36.06],[621.11,1038.9,37.48],[1287.44,1604.81,38.94],[1132.94,1800.02,40.22],[794.22,3462.72,42.88],[2820.4,3789.04,33.28],[2786.35,4734.36,44.96],[565.22,4774.84,25.14],[986.09,6311.95,40.1],[2927.57,5217.91,40.48],[336.62,357.76,43.66],[4623.38,365.11,42.48],[486.16,5766.56,43.3],[754.85,3476.06,33.82],[1303.74,1711.19,28],[1949.96,1098.31,22.16],[3342.56,2062.81,21.44],[3793.26,3024.19,32.72],[3563.37,3394.67,18.8],[907.62,3799.33,38.86]]
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
from PIL import Image

import csv

# cmap = cm.get_cmap("RdYlGn")
# norm = mcolors.Normalize(vmin=.3, vmax=.87)

plt.imshow(Image.open(im))

nms_radius_px = 60

with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    points = [row for row in reader]
    for i, point in enumerate(points):
        
        
        if i == 0:
            continue  
        
        point = [float(point[0]), float(point[1]), float(point[2])]
        if point[2] > .5:
            rect = patches.Rectangle((point[0] - nms_radius_px, point[1] - nms_radius_px), 2 * nms_radius_px, 2 * nms_radius_px, linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
            plt.scatter(point[0], point[1], c='g', s=2)

plt.axis('off')
plt.show()

# for p in og_points:
#     plt.scatter(p[0], p[1], c='r', s=5)

# fps = [(4075.1201171875, 1402.47998046875), (95.5199966430664, 4945.259765625), (614.7000122070312, 516.4299926757812), (1964.47998046875, 4255.0400390625), (1781.52001953125, 5624.9599609375)]
# for fp in fps:
#     plt.scatter(fp[0], fp[1], c='b', s=2)


# # plt.show()
# import pprint
# from lesion_dinov3.utils import nms_points

# pts = np.load("debug_pts.npy")
# scores = np.load("debug_scores.npy")

# pts, scores = nms_points(pts, scores, radius=60)

# pts_index = np.where(scores > 0.5)[0]

# scores = scores[pts_index]
# pts = pts[pts_index]

# nms_radius_px = 60

# plt.imshow(Image.open(im))
# # for i, (point, score) in enumerate(zip(pts, scores)):
# #     plt.scatter(point[0], point[1], c='g', s=2)
#         #plt.scatter(point[0], point[1], c=cmap(norm(score)), s=2)

# # for point in og_points:
# #     plt.scatter(point[0], point[1], c='r', s=2)

# pts, scores = nms_points(pts, scores, radius=60)
# crops = []
# for i, point in enumerate(pts):
#     # for p in og_points:
#     #     if np.sqrt(((point[:-1] - p) ** 2).sum()) < nms_radius_px:
#     #         continue
#     #     else:
#     #         crop = [point[0] - nms_radius_px, point[1] - nms_radius_px, point[0] + nms_radius_px, point[1] + nms_radius_px]
#     #         image_crop = Image.open(im).crop(crop)
#     #         crops.append(image_crop)
#     #         image_crop.save(f"crops/crop_{i}.png")
#     rect = patches.Rectangle((point[0] - nms_radius_px, point[1] - nms_radius_px), 2 * nms_radius_px, 2 * nms_radius_px, linewidth=1, edgecolor='g', facecolor='none')
#     plt.gca().add_patch(rect)

# # dists = []
# # order = np.argsort(scores)[::-1]
# # for i in order:
# #     curr_point = pts[i]
# #     score = scores[i]
# #     if score < 0.5:
# #         continue
# #     # print(f"Point: {point}, Score: {score}")
# #     for point in pts:
# #         if np.array_equal(curr_point, point):
# #             continue
# #         d = np.sqrt(((curr_point - point) ** 2).sum())
# #         dists.append([d, curr_point, point])
# #         #print(f"Point {curr_point} is {d} pixels from point {point}.")
# # dists.sort( key=lambda x: x[0])
# # print(len(dists))
# # for d, p1, p2 in dists:
# #     if d < 200:
# #         plt.scatter(p1[0], p1[1], c='g', s=10)
# #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', linewidth=0.5)
            
# # print(f"Loaded {len(pts)} points and {len(scores)} scores from debug files.")

# plt.axis('off')

# plt.show()
