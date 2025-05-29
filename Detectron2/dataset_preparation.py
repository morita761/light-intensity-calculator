import cv2
import numpy as np
import json
import os
from PIL import Image
import glob

def extract_blue_mask(mask_path):
    mask = cv2.imread(mask_path)
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # 青色の範囲を定義
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return blue_mask

def create_coco_annotations(img_dir, mask_dir, output_json_path):
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 0,
            "name": "horseshoe"
        }]
    }

    annotation_id = 0
    for image_id, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        mask_path = os.path.join(mask_dir, base_name + "_mask.tif")

        if not os.path.exists(mask_path):
            print(f"⚠️ マスクが見つかりません: {mask_path}")
            continue

        # 画像サイズ
        image = Image.open(image_path)
        width, height = image.size

        coco_dict["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # マスク処理
        mask = extract_blue_mask(mask_path)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            segmentation = contour.flatten().tolist()
            if len(segmentation) < 3:  # 少なすぎるポリゴンは無視
                continue

            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "segmentation": [segmentation],
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

    # 書き出し
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"✅ アノテーション生成完了: {output_json_path}")

# 使用例
create_coco_annotations(
    img_dir="../data/img",
    mask_dir="../data/masks",
    output_json_path="train_annotations_horseshoe.json"
)

# create_coco_annotation("simpledataset/images/003.png", "simpledataset/masks/003_mask.png", "train_annotations.json")

