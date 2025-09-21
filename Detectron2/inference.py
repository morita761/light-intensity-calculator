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
import func.intensityCalc as calc
import sys

# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
cfg.MODEL.DEVICE = "cuda"
# cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.2

# --- 推論器 ---
predictor = DefaultPredictor(cfg)

# --- 入力ファイル ---
input_path = "../data/img/up_Fz_green_stronger.tif"
# input_path = "../data/25cas9P/01.tif"

# --- 入力ファイル名やIDを抽出 ---
base_name = os.path.splitext(os.path.basename(input_path))[0]  # '003'

# --- 出力ディレクトリの準備 ---
output_dir = f"./output/test3/{base_name}/"
os.makedirs(output_dir, exist_ok=True)
kmeans_dir = f"{output_dir}/kmeans/"
os.makedirs(kmeans_dir, exist_ok=True)

# === 出力ファイル名の定義 ===
csv_path = os.path.join(output_dir, f"evaluation_results_{base_name}_thresh.csv")
annotated_img_path = os.path.join(output_dir, f"annotated_result_horseshoe_{base_name}_thresh.png")
visualized_img_path = os.path.join(output_dir, f"result_horseshoe_{base_name}_thresh.png")
# --- 画像読み込み ---
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError("画像が読み込めませんでした")

cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
# 読み込んだ画像の高さと幅を取得，整数で割り算
height = image.shape[0] //2
width = image.shape[1] //2
cv2.resizeWindow('Sample Image', width, height)

# --- 推論 ---
outputs = predictor(image)

# 結果保存用リスト
results = []

# インスタンス情報を取得
instances = outputs["instances"].to("cpu")
masks = instances.pred_masks.numpy()
boxes = instances.pred_boxes.tensor.numpy()

# for i, (mask, box) in enumerate(zip(masks, boxes)):
#     x1, y1, x2, y2 = box.astype(int)
#     h = mask.shape[0]

#     # --- 下部開放チェック
#     lower_quarter = mask[int(h * 0.75):, :]
#     cols_with_mask = np.any(lower_quarter, axis=0)
#     is_open = not (cols_with_mask[0] and cols_with_mask[-1])

#     # --- 面積計算
#     area = np.sum(mask)
#     if area < 100:
#         continue

#     # 画像からbox領域をトリミング
#     cropped_image = image[y1:y2, x1:x2]

    # # 緑色ポイントの抽出
    # points, green_mask = calc.extract_green_points(cropped_image)    

    # center, direction = calc.perform_pca(points)
    # labels, cluster_centers = calc.perform_kmeans(points)
    # result_img, left_cluster, right_cluster, left_brightness, right_brightness = calc.draw_results(cropped_image, center, direction, cluster_centers)
    # filename = f"{kmeans_dir}/pca_kmeans_result_{i}.png"
    # cv2.imwrite(filename, result_img)

    # # --- 結果リストに追加
    # results.append({
    #     "id": i,
    #     "is_open": is_open,
    #     "area": area,
    #     "left_green": "{:.3f}".format(left_brightness),
    #     "right_green": "{:.3f}".format(right_brightness)
    # })

# --- 結果をCSVに保存
# os.makedirs("output", exist_ok=True)
# df = pd.DataFrame(results)
# df.to_csv(csv_path, index=False)
# print(f"✅ 結果を {csv_path} に保存しました")

# --- 画像保存\
# cv2.imwrite(annotated_img_path, image)
# print(f"✅ 評価付き画像を {annotated_img_path} に保存しました")

# --- メタデータ登録 ---
MetadataCatalog.get("horseshoe_dataset").set(thing_classes=["horseshoe"])

# --- 可視化と保存 ---
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("horseshoe_dataset"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite(visualized_img_path, out.get_image()[:, :, ::-1])
cv2.imshow("Sample Image", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
print(f"✅ 可視化画像を {visualized_img_path} に保存しました")

cv2.destroyAllWindows()