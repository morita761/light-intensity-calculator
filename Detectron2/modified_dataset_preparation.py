import cv2
import numpy as np
import json
import os
from PIL import Image
import glob

# アノテーションに使用する色のHSV範囲を定義
# 各色のHue（色相）範囲を厳密に設定することが重要
# Saturation (彩度) と Value (明度) は広めに取るのが一般的ですが、ノイズが入る場合は調整
ANNOTATION_COLORS_HSV = {
    "red": {
        "lower": np.array([0, 100, 100]),   # Hueが0-10, 170-180の間
        "upper": np.array([10, 255, 255]),
        "lower2": np.array([171, 100, 100]), # 赤はHueが両端にあるため2つの範囲が必要
        "upper2": np.array([180, 255, 255])
    },
    "blue": {
        "lower": np.array([101, 100, 100]), # Hueが100-130くらい
        "upper": np.array([130, 255, 255])
    },
    "cyan": {
        "lower": np.array([90, 100, 100]), # Hueが85-95くらい
        "upper": np.array([100, 255, 255])
    },
    "magenta": {
        "lower": np.array([160, 100, 100]), # Hueが140-160くらい
        "upper": np.array([170, 255, 255])
    }
    # 他の色を追加する場合はここに追加
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

def create_coco_annotations(img_dir, mask_dir, output_json_path):
    # image_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 0,
            "name": "horseshoe"
        }] # 現状はhorseshoeクラスのみなのでIDは0
    }

    annotation_id = 0
    for image_id, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        # mask_path = os.path.join(mask_dir, base_name + "_mask.tif") # アノテーションマスクファイル名
        mask_path = os.path.join(mask_dir, base_name + "_mask.png") # アノテーションマスクファイル名

        if not os.path.exists(mask_path):
            print(f"⚠️ マスクが見つかりません: {mask_path}")
            continue

        # オリジナル画像サイズを取得
        original_image = Image.open(image_path)
        width, height = original_image.size

        coco_dict["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # アノテーションマスク画像を読み込む (カラーで読み込む必要あり)
        annotation_mask_bgr = cv2.imread(mask_path)
        if annotation_mask_bgr is None:
            print(f"❌ アノテーションマスク画像が読み込めませんでした: {mask_path}")
            continue

        # 定義された各色ごとに処理
        for color_name in ANNOTATION_COLORS_HSV.keys():
            # 特定の色（インスタンス）のマスクを抽出
            instance_mask = extract_color_mask(annotation_mask_bgr, color_name)
            # cv2.imshow('Sample Image', instance_mask)
            # cv2.waitKey(2*1000)

            # 各インスタンスマスクから輪郭を検出
            # RETR_EXTERNAL で外側の輪郭のみを検出（各色の塊ごとに）
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):
                segmentation = contour.flatten().tolist()
                # ポリゴン点が少なすぎる場合は無視（ノイズ対策）
                if len(segmentation) < 6: # 少なくとも3点必要だが、より厳しく6点以上とすることも
                    continue
                if cv2.contourArea(contour) < 20:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                # 最初に描画したマスクから膨張処理を開始
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                rel_contour = contour - np.array([x, y])
                cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)
                
                kernel = np.ones((3, 3), np.uint8)
                iterations = 0
                contour_mask_local = temp_mask.copy()
    
                while True:
                    # 占有率を計算
                    total_pixels = w * h
                    white_pixels = cv2.countNonZero(contour_mask_local)
                    occupancy_rate = white_pixels / total_pixels if total_pixels > 0 else 0
                    # print(f"Iteration: {iterations}, Occupancy Rate: {occupancy_rate:.2f}")
    
                    # 占有率が60%を超えるか、試行回数が4回を超えたらループを抜ける
                    if occupancy_rate > 0.60 or iterations >= 4:
                        # if iterations >=1: 
                            # cv2.imshow("ERROR Mask", contour_mask_local)
                            # cv2.waitKey(0)
                        break
                    
                    # 輪郭をさらに膨張させる
                    contour_mask_local = cv2.dilate(contour_mask_local, kernel, iterations=1)
                    iterations += 1
                    
                
                if(iterations >2): # 膨張が4のときはスキップする
                    # cv2.imshow("skip Mask", contour_mask_local)
                    # cv2.waitKey(0)
                    continue
                color = (0, 255, 255) # 黄色 (BGR) for ignored contours
                # 最終的な輪郭を取得し、debug_imgに描画
                final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if final_contours:
                    # 最初の輪郭（最大のものと仮定）を描画
                    final_absolute_contour = final_contours[0] + np.array([x, y])
                    cv2.drawContours(annotation_mask_bgr, [final_absolute_contour], -1, color, -1)
                    area = cv2.contourArea(final_absolute_contour)

                coco_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0, # 全ての馬蹄形は同じカテゴリID 0
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0 # 各インスタンスは個別のオブジェクトなので0
                })
                annotation_id += 1
            
        # debug
        cv2.imshow('Sample Image', annotation_mask_bgr)
        cv2.waitKey(0)

    # 書き出し
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"✅ アノテーション生成完了: {output_json_path}")

# 使用例
# アノテーションマスクは、個別の馬蹄形が異なる色で塗られた単一の画像ファイル
# 例: ../data/masks/001_mask.tif の中に、赤、青、シアン、マゼンタの馬蹄形が描かれている
create_coco_annotations(
    img_dir="../data/2img",
    mask_dir="../data/2masks", # ここに色分けされたマスク画像があることを想定
    output_json_path="train_annotations_horseshoe_RGB.json"
)

create_coco_annotations(
    img_dir="../data/2img_val",
    mask_dir="../data/2masks_val",
    output_json_path="val_annotations_horseshoe_RGB.json"
)