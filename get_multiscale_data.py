import pprint
import os
import pandas as pd
import json
import random
import math

AMIN_1 = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_images_1'
AMIN_1_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_improved_annotations_1"
AMIN_2 = 'C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_images_2'
AMIN_2_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/amin_prescan_improved_annotations_2"
PATIENT_1 = 'C:/Users/davoo/Development/LumoLesionClassifier/patient_images'
PATIENT_1_ANNOTATION = "C:/Users/davoo/Development/LumoLesionClassifier/patient_improved_annotations"

def walk_dir(dir):
    files = []
    for subdir in os.listdir(dir):
        for file in os.listdir(os.path.join(dir, subdir)):
            files.append(os.path.join(dir,subdir,file))
    
    return files

image_folders = [
    [AMIN_1, AMIN_2, PATIENT_1]
    ] + [
        [
            AMIN_1 + f"_downsample_{i}x", 
            AMIN_2 + f"_downsample_{i}x",
            PATIENT_1 + f"_downsample_{i}x"
        ] for i in [2,4,8]]
annotation_folders = [AMIN_1_ANNOTATION, AMIN_2_ANNOTATION, PATIENT_1_ANNOTATION]
annotation_files = [walk_dir(folder) for folder in annotation_folders]




data = []
image_files = [
    [walk_dir(dir) for dir in dirs] for dirs in image_folders
]

data = []
for image in range(40):
    for scan in range(3):
        row = []
        for scale in range(4):
            img = image_files[scale][scan][image]
            row.append(img)
        annotation_file = annotation_files[scan][image]
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
            points = [[ann['x']  , ann['y'] , ann['radius'] * 2] for ann in annotation['image']['points']]
            
            csv_value = json.dumps(points, separators=(",", ":"))
            row.append(csv_value)
        data.append(row)

pprint.pprint(data[0])

random.shuffle(data)
index = math.floor(len(data) * 0.8)

train_data = data[:index]
val_data = data[index:]

col_names = [
    "image_path_1", "image_path_2", "image_path_4", "image_path_8", "points_json"
    ]

train_df = pd.DataFrame(train_data, columns=col_names)

val_df =  pd.DataFrame(val_data, columns=col_names)
val_df = val_df[["image_path_1", "points_json"]]
val_df = val_df.rename(columns = {'image_path_1' : 'image_path'})
val_df['prior_path'] = ""

train_df.to_csv("data_multiscale/train.csv", index=False)
val_df.to_csv("data_multiscale/val.csv", index=False)




        

