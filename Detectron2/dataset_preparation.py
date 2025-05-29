import cv2
import numpy as np
import json
import os
from PIL import Image

def extract_blue_mask(mask_path):
    mask = cv2.imread(mask_path)
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # 青色の範囲を定義
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return blue_mask

def create_coco_annotation(image_path, mask_path, output_json_path):
    mask = extract_blue_mask(mask_path)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image = Image.open(image_path)
    width, height = image.size

    annotations = []
    for i, contour in enumerate(contours):
        segmentation = contour.flatten().tolist()
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        annotations.append({
            "id": i,
            "image_id": 1,
            "category_id": 0,
            "segmentation": [segmentation],
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        })

    coco_dict = {
        "images": [{
            "id": 1,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }],
        "annotations": annotations,
        "categories": [{
            "id": 0,
            "name": "horseshoe"
        }]
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

# create_coco_annotation("simpledataset/images/003.png", "simpledataset/masks/003_mask.png", "train_annotations.json")
create_coco_annotation("../data/img/003.tif", "../data/masks/003_mask.tif", "train_annotations_horseshoe.json")
