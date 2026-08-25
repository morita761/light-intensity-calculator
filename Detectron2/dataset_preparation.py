import cv2
import numpy as np
import json
import os
from PIL import Image
import glob

from func.cli_errors import require_dir, run_main


# アノテーションに使用する色のHSV範囲を定義
ANNOTATION_COLORS_HSV = {
    "red": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255]),
        "lower2": np.array([170, 100, 100]),
        "upper2": np.array([180, 255, 255])
    },
    "blue": {
        "lower": np.array([100, 100, 100]),
        "upper": np.array([130, 255, 255])
    },
    "cyan": {
        "lower": np.array([85, 100, 100]),
        "upper": np.array([100, 255, 255])
    },
    "magenta": {
        "lower": np.array([140, 100, 100]),
        "upper": np.array([170, 255, 255])
    }
}

def extract_color_mask(mask_image_bgr, color_name):
    """指定された色のHSV範囲に基づいてマスクを抽出する"""
    hsv = cv2.cvtColor(mask_image_bgr, cv2.COLOR_BGR2HSV)
    color_ranges = ANNOTATION_COLORS_HSV[color_name]

    mask1 = cv2.inRange(hsv, color_ranges["lower"], color_ranges["upper"])
    if "lower2" in color_ranges: # 赤のようにHueが2つの範囲にまたがる場合
        mask2 = cv2.inRange(hsv, color_ranges["lower2"], color_ranges["upper2"])
        final_mask = cv2.bitwise_or(mask1, mask2)
    else:
        final_mask = mask1
    return final_mask

def extract_color_masks(mask_path):
    """
    マスク画像から色別にマスクを抽出

    - blue, cyan → horseshoe (category_id: 0)
    - magenta → negative (category_id: 1)

    Returns:
        list of (mask, category_id) tuples
    """
    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None:
        print(f"⚠️ マスク画像を読み込めません: {mask_path}")
        return []

    masks_with_categories = []

    # 色名 → category_id の対応
    color_to_category = {
        "red": 1,       # negative
        "blue": 0,      # horseshoe
        "cyan": 0,      # horseshoe
        "magenta": 0,   # horseshoe
    }

    for color_name, category_id in color_to_category.items():
        color_mask = extract_color_mask(mask_bgr, color_name)

        if np.any(color_mask):
            masks_with_categories.append((color_mask, category_id))

    return masks_with_categories


def extract_blue_mask(mask_path):
    """後方互換性のため残す（青色マスクのみ抽出）"""
    mask = cv2.imread(mask_path)
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return blue_mask


def create_coco_annotations(img_dir, mask_dir, output_json_path, use_multicolor=True):
    """
    COCOフォーマットのアノテーションを生成

    Args:
        img_dir: 画像ディレクトリ
        mask_dir: マスクディレクトリ
        output_json_path: 出力JSONパス
        use_multicolor: True=複数色対応（赤=negative, 他=horseshoe）, False=青のみ
    """
    require_dir(img_dir, "img_dir（画像ディレクトリ）")
    require_dir(mask_dir, "mask_dir（マスクディレクトリ）")

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))

    # .tifがない場合は.pngも探す
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "horseshoe"},
            {"id": 1, "name": "negative"}
        ]
    }

    annotation_id = 0
    horseshoe_count = 0
    negative_count = 0

    for image_id, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]

        # マスクファイルを探す（.tifと.pngの両方を試す）
        mask_path = None
        for ext in [".tif", ".png"]:
            candidate = os.path.join(mask_dir, base_name + "_mask" + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"⚠️ マスクが見つかりません: {base_name}_mask.*")
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

        if use_multicolor:
            # 複数色対応
            masks_with_categories = extract_color_masks(mask_path)

            for color_mask, category_id in masks_with_categories:
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    segmentation = contour.flatten().tolist()
                    if len(segmentation) < 6:  # 最低3点必要
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)

                    if area < 10:  # 小さすぎるものは除外
                        continue

                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [segmentation],
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1

                    if category_id == 0:
                        horseshoe_count += 1
                    else:
                        negative_count += 1
        else:
            # 青色のみ（従来動作）
            mask = extract_blue_mask(mask_path)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                segmentation = contour.flatten().tolist()
                if len(segmentation) < 6:
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
                horseshoe_count += 1

    # 書き出し
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ アノテーション生成完了: {output_json_path}")
    print(f"   画像数: {len(coco_dict['images'])}")
    print(f"   horseshoe: {horseshoe_count}個")
    print(f"   negative: {negative_count}個")


def main():
    # 学習データ（複数色対応）
    create_coco_annotations(
        img_dir="../data/train/images",
        mask_dir="../data/train/masks",
        output_json_path="train_annotations.json",
        use_multicolor=True
    )

    # 検証データ（複数色対応）
    create_coco_annotations(
        img_dir="../data/val/images",
        mask_dir="../data/val/masks",
        output_json_path="val_annotations.json",
        use_multicolor=True
    )


if __name__ == "__main__":
    run_main(main)
