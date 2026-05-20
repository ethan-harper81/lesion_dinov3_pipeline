import csv
import os
import json
from pprint import pprint
import random
import math

AMIN_1 = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_images_1'
AMIN_1_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_improved_annotations_1"
AMIN_2 = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_images_2'
AMIN_2_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_improved_annotations_2"
PATIENT_1 = 'C:/Users/davoo/Development/LumoLesionClassifier/patient_images'
PATIENT_1_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/patient_improved_annotations"

sample_factors = [1, 2, 4, 8]

hold_out = True # Holds out image C:/Users/davoo/Development/LumoLesionClassifier/patient_images/front/wide_00_001.jpg

folders = [
    [AMIN_1 , AMIN_1_ANNOTATION], 
    [AMIN_2, AMIN_2_ANNOTATION], 
    [PATIENT_1, PATIENT_1_ANNOTATION]
    ]

data = []
save_file = ''
save_ann = ''
i = 0
for im_dir, annotation_dir in folders:
    fs = 0
    i+=1
    for subdir in os.listdir(im_dir):
        for file in os.listdir(os.path.join(im_dir, subdir)):
            fs += 1
            annotation_file = os.path.join(annotation_dir, subdir, file.split('.')[0] + '-points.json')
            if im_dir == PATIENT_1 and subdir == "front" and file == "wide_00_001.jpg" and hold_out:
                print("saving file", os.path.join(im_dir, subdir, file))
                save_file = os.path.join(im_dir, subdir, file)
                save_ann = annotation_file
                continue

            

            if not os.path.exists(annotation_file):
                print(annotation_file, "does not exist, skipping")
                data.append([os.path.join(im_dir, subdir, file), '[]', ''])
                continue
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
                points = [[ann['x']  , ann['y'] , ann['radius'] * 2] for ann in annotation['image']['points']]
                
                csv_value = json.dumps(points, separators=(",", ":"))
                data.append([os.path.join(im_dir, subdir, file), csv_value, ''])
            
    print("extracted ", fs, " files")

random.shuffle(data)

index = math.floor(len(data) * .7)
index_2 = math.floor(len(data) * .85)
train_data = data[:index]
print(len(train_data))
val_data = data[index:index_2]
print(len(val_data))
test_data = data[index_2:]
print(test_data[0])

if hold_out:
    with open(save_ann, 'r') as f:
        annotation = json.load(f)
        points = [[ann['x']  , ann['y'] , ann['radius'] * 2] for ann in annotation['image']['points']]
        
        csv_value = json.dumps(points, separators=(",", ":"))
        test_data.append([save_file, csv_value, ''])

print(len(test_data))

for sample_factor in sample_factors:


    train_data_sample = [[d[0].split('\\')[0] + f'_downsample_{sample_factor}x' + "/" + d[0].split('\\')[1] + "/" + d[0].split('\\')[2], json.dumps([[p[0] / sample_factor , p[1] / sample_factor, p[2] / sample_factor] for p in json.loads(d[1])], separators=(",", ":")), d[2] ] for d in train_data ] if sample_factor != 1 else train_data
    val_data_sample = [[d[0].split('\\')[0] + f'_downsample_{sample_factor}x' + "/" + d[0].split('\\')[1] + "/" + d[0].split('\\')[2], json.dumps([[p[0] / sample_factor, p[1] / sample_factor, p[2] / sample_factor] for p in json.loads(d[1])], separators=(",", ":")), d[2] ] for d in val_data ] if sample_factor != 1 else val_data
    test_data_sample = [[d[0].split('\\')[0] + f'_downsample_{sample_factor}x' + "/" + d[0].split('\\')[1] + "/" + d[0].split('\\')[2], json.dumps([[p[0] / sample_factor, p[1] / sample_factor, p[2] / sample_factor] for p in json.loads(d[1])], separators=(",", ":")), d[2] ] for d in test_data ] if sample_factor != 1 else test_data

    data_folder = f'data_{sample_factor}x' if sample_factor != 1 else 'data'

    with open(f'{data_folder}/train.csv', "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "points_json", "prior_path"])
        writer.writerows(train_data_sample)

    with open(f'{data_folder}/val.csv', "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "points_json", "prior_path"])
        writer.writerows(val_data_sample)

    with open(f'{data_folder}/test.csv', "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "points_json", "prior_path"])
        writer.writerows(test_data_sample)