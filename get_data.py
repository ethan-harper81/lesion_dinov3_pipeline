import csv
import os
import json
from pprint import pprint
import random
import math

im_dir = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_images_1'
im_dir_ANNOTATION = "C:/Users/davoo/Development/LumoServer/data/amin_data/annotation_data_first"
AMIN_2 = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_images_2'
AMIN_2_ANNOTATION = "C:/Users/davoo/Development/LumoServer/data/amin_data/annotation_data_second"
PATIENT_1 = 'C:/Users/davoo/Development/LumoLesionClassifier/patient_images'
PATIENT_1_ANNOTATION = "C:/Users/davoo/Development/LumoServer/data/original_annotation_data"

folders = [
    [im_dir, im_dir_ANNOTATION], 
    [AMIN_2, AMIN_2_ANNOTATION], 
    [PATIENT_1, PATIENT_1_ANNOTATION]
    ]


data = []
for im_dir, annotation_dir in folders:
    # print(im_dir)
    # print(annotation_dir)
    for subdir in os.listdir(im_dir):
        for file in os.listdir(os.path.join(im_dir, subdir)):
            annotation_file = os.path.join(annotation_dir, subdir, file.split('.')[0] + '.json')
            if not os.path.exists(annotation_file):
                data.append([os.path.join(im_dir, subdir, file), '[]', ''])
                continue
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
                points = [[ann['x'] * 8, ann['y'] * 8] for ann in annotation['annotations']]
                
                csv_value = json.dumps(points, separators=(",", ":"))
                data.append([os.path.join(im_dir, subdir, file), csv_value, ''])

random.shuffle(data)

index = math.floor(len(data) * .7)
index_2 = math.floor(len(data) * .85)
train_data = data[:index]
val_data = data[index:index_2]
test_data = data[index_2:]

with open('data/train.csv', "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "points_json", "prior_path"])
    writer.writerows(train_data)

with open('data/val.csv', "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "points_json", "prior_path"])
    writer.writerows(val_data)

with open('data/test.csv', "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "points_json", "prior_path"])
    writer.writerows(test_data)