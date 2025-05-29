from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import torch
import numpy as np
import pandas as pd
import os
import func.splitobje as split
import sys

# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

# --- 推論器 ---
predictor = DefaultPredictor(cfg)

# --- 画像読み込み ---
# image = cv2.imread("./simpledataset/images/01.png")
image = cv2.imread("../data/img/003.tif")
if image is None:
    raise FileNotFoundError("画像が読み込めませんでした")

cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
# 読み込んだ画像の高さと幅を取得，整数で割り算
height = image.shape[0] //2
width = image.shape[1] //2

# ウィンドウサイズを変更，cv2.destroyAllWindows()をするまで使用できる
cv2.resizeWindow('Sample Image', width, height)

green_mask = split.extract_green_object(image)
cv2.imshow('Sample Image', green_mask)
cv2.waitKey(0)

left_mask, right_mask = split.split_left_right(green_mask)
cv2.imshow('Sample Image', left_mask)
cv2.waitKey(0) 
cv2.imshow('Sample Image', right_mask)
# cv2.imshow('Sample Image',image)
cv2.waitKey()
cv2.destroyAllWindows()

# sys.exit()

# --- 推論 ---
outputs = predictor(left_mask)

# 結果保存用リスト
results = []

# インスタンス情報を取得
instances = outputs["instances"].to("cpu")
masks = instances.pred_masks.numpy()
boxes = instances.pred_boxes.tensor.numpy()

for i, (mask, box) in enumerate(zip(masks, boxes)):
    x1, y1, x2, y2 = box.astype(int)
    h = mask.shape[0]

    # --- 下部開放チェック
    lower_quarter = mask[int(h * 0.75):, :]
    cols_with_mask = np.any(lower_quarter, axis=0)
    is_open = not (cols_with_mask[0] and cols_with_mask[-1])

    # --- 面積計算
    area = np.sum(mask)

    # --- 緑輝度計算
    # --- 緑輝度計算用パッチ座標
    green_thresh = 60
    patch_size = 20
    pad = 2
    left_x1 = x1 + pad
    left_x2 = x1 + patch_size + pad
    right_x1 = x2 - patch_size - pad
    right_x2 = x2 - pad
    y_patch1 = y2 - patch_size - 2
    y_patch2 = y2 - 2
    
    # --- パッチ抽出
    left_patch = image[y_patch1:y_patch2, left_x1:left_x2, 1]
    right_patch = image[y_patch1:y_patch2, right_x1:right_x2, 1]

    # --- 緑輝度計算（しきい値以上のみ） ---
    left_mean = np.mean(left_patch[left_patch > green_thresh]) if left_patch.size > 0 else 0
    right_mean = np.mean(right_patch[right_patch > green_thresh]) if right_patch.size > 0 else 0
    
    # --- パッチ領域を矩形で描画
    cv2.rectangle(image, (left_x1, y_patch1), (left_x2, y_patch2), (0, 255, 255), 1)
    cv2.rectangle(image, (right_x1, y_patch1), (right_x2, y_patch2), (0, 255, 255), 1)

    # --- 結果リストに追加
    results.append({
        "id": i,
        "is_open": is_open,
        "area": area,
        "left_green": round(left_mean, 1),
        "right_green": round(right_mean, 1)
    })

    # --- 画像に描画
    text = f"#{i} Area:{area} Open:{is_open}\nG(L):{left_mean:.1f} G(R):{right_mean:.1f}"
    cv2.putText(image, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# --- 結果をCSVに保存
os.makedirs("output", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("output/evaluation_results_thresh.csv", index=False)
print("✅ 結果を output/evaluation_results_thresh.csv に保存しました")

# --- 画像保存
cv2.imwrite("output/annotated_result_horseshoe_003_thresh.png", image)
print("✅ 評価付き画像を output/annotated_result_horseshoe_003_thresh.png に保存しました")


# --- メタデータ登録 ---
MetadataCatalog.get("horseshoe_dataset").set(thing_classes=["horseshoe"])

# --- 可視化と保存 ---
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("horseshoe_dataset"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output/result_horseshoe_003_thresh.png", out.get_image()[:, :, ::-1])
