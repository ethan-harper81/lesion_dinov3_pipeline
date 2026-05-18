# Testing model training for each of 1x, 2x, 4x, 8x downsamples of original image size
# Output for image C:/Users/davoo/Development/LumoLesionClassifier/patient_images\front\wide_00_001.jpg
import timeit
import yaml
import csv
import json

from train import main as train
from infer_full_image import main as infer
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiprocessing import freeze_support

import os

def save_plot(im, hp, cp, pnts, out_dir):

    im = Image.open(im)
    points = []

    with open(cp, mode='r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue

            points.append((float(row[0]), float(row[1]), float(row[2])))

    heatmap = np.load(hp)

    plt.imshow(im)
    # /plt.imshow(heatmap, alpha=0.5, cmap="jet")
    for point in points:

        rect = patches.Rectangle((point[0] - 5, point[1] - 5), 10, 10, linewidth=1, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)

    og_points = pnts
    print(str(len(og_points)) + "og points vs" + str(len(points)) + "predicted points")

    for point in og_points:
        rect = patches.Rectangle((point[0] - 1, point[1] - 1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.savefig(os.path.join(out_dir, "test_inference.jpg"))
    plt.close()

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

configs = [
    [9504, 6336, 768, 512],
    [4752, 3168, 624, 416],
    [2376, 1584, 432, 288],
    [1188, 792, 225, 150]
]

sample_factors = {
    "8x": [1188, 792, 225, 150],
    "4x": [2376, 1584, 432, 288],
    "2x": [4752, 3168, 624, 416],
    "1x": [9504, 6336, 768, 512]
}

def seconds_to_min_sec(total_seconds):
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{int(minutes)}:{int(seconds)}"

def main():
    cfg = load_cfg("config.yaml")

    og_cfg = cfg

    image = 'C:/Users/davoo/Development/LumoLesionClassifier/patient_images-/front/wide_00_001.jpg'

    points_f = "C:/Users/davoo/Development/LumoServer/data/original_annotation_data/front/wide_00_001.json"

    for factor, data in sample_factors.items():

        fac = int(factor[0])

        with open(points_f, 'r') as f:
            annotation = json.load(f)
            points = [[ann['x'] * int(8 /fac) , ann['y'] * int(8 /fac)] for ann in annotation['annotations']]

        checkpoint_folder = f"runs/lesion_dinov3_vits_{factor}_downsample"

        checkpoint = checkpoint_folder + "/best.pt"

        img = image.split('-')[0] + f"_downsample_{factor}" + image.split('-')[1] if factor != '1x' else image.split('-')[0] + image.split('-')[1]

        print(f'---{factor}---')
        cfg["data"]["tile_size"] = data[2]
        cfg['infer']['tile_size'] = data[2]
        cfg['infer']['stride'] = data[3]
        cfg['train']['output_dir'] = checkpoint_folder
        cfg['infer']['checkpoint'] = checkpoint

        print(f'---Traing on Factor: {factor}---')
        t0 = timeit.default_timer()
        train(cfg)
        t1 = timeit.default_timer()
        print(f"Training complete in {seconds_to_min_sec(t1-t0)}")
        print("---Inference On single Image---")
        hp, pp, cp = infer(checkpoint, img, fac)

        save_plot(img, hp, cp, points, checkpoint_folder)
        t2 = timeit.default_timer()
        print(f"Inference complete in {seconds_to_min_sec(t2-t1)}")

if __name__ == "__main__":
    freeze_support()
    main()


        
