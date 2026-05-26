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
from PIL import Image

import csv

# cmap = cm.get_cmap("RdYlGn")
# norm = mcolors.Normalize(vmin=.3, vmax=.87)

plt.imshow(Image.open(im))

nms_radius_px = 60

num_candidates = 25

with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    points = [row for row in reader]
    total = 0
    for i, point in enumerate(points):
        
        if i == num_candidates + 1:
            break
        
        if i == 0:
            continue  
        
        point = [float(point[0]), float(point[1]), float(point[2])]
        if point[2] > .5:
            total += 1
            rect = patches.Rectangle((point[0] - nms_radius_px, point[1] - nms_radius_px), 2 * nms_radius_px, 2 * nms_radius_px, linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
            #plt.scatter(point[0], point[1], c='g', s=2)
        # else:
        #     rect = patches.Rectangle((point[0] - nms_radius_px, point[1] - nms_radius_px), 2 * nms_radius_px, 2 * nms_radius_px, linewidth=1, edgecolor='y', facecolor='none')
        #     plt.gca().add_patch(rect)
            #plt.scatter(point[0], point[1], c='r', s=2)
# plt.title(f"Total candidates > 0.5: {total / len(points) * 100:.2f}% of {len(points)}")
plt.axis('off')
# plt.show()

for p in og_points:
    plt.scatter(p[0], p[1], c='r', s=5)

# fps = [(4075.1201171875, 1402.47998046875), (95.5199966430664, 4945.259765625), (614.7000122070312, 516.4299926757812), (1964.47998046875, 4255.0400390625), (1781.52001953125, 5624.9599609375)]
# for fp in fps:
#     plt.scatter(fp[0], fp[1], c='b', s=2)


plt.show()
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