from detectron2.utils.visualizer import Visualizer
import numpy as np
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
from detectron2.structures import Instances, Boxes, BitMasks

# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

# --- 推論器 ---
predictor = DefaultPredictor(cfg)

# --- 入力ファイル ---
input_path = "../data/img/003.tif"
direction = "merge" # l or r

# --- 入力ファイル名やIDを抽出 ---
base_name = os.path.splitext(os.path.basename(input_path))[0]  # '003'

# --- 出力ディレクトリの準備 ---
output_dir = f"./output/controlFzGFP/{base_name}/{direction}/"
os.makedirs(output_dir, exist_ok=True)
if(direction != " "):
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

green_mask = split.extract_green_object(image, coefficient_left=0.10, coefficient_right=0.80, coefficient_bottom=0.9, coefficient_top=0.12)
cv2.imshow('Sample Image', green_mask)
cv2.waitKey(10*1000)

left_mask, right_mask = split.split_left_right(green_mask)

# 左右の出力を取得
outputs_left = predictor(left_mask)
outputs_right = predictor(right_mask)

# 左右のインスタンスを統合（CPUに移して）
instances_left = outputs_left["instances"].to("cpu")
instances_right = outputs_right["instances"].to("cpu")

# --- 元画像サイズに合わせた統合インスタンス作成
merged_instances = Instances(image.shape[:2])

# pred_boxes
boxes_left = instances_left.pred_boxes.tensor
boxes_right = instances_right.pred_boxes.tensor
merged_instances.pred_boxes = Boxes(torch.cat([boxes_left, boxes_right], dim=0))

# scores
scores_left = instances_left.scores
scores_right = instances_right.scores
merged_instances.scores = torch.cat([scores_left, scores_right], dim=0)

# pred_classes
classes_left = instances_left.pred_classes
classes_right = instances_right.pred_classes
merged_instances.pred_classes = torch.cat([classes_left, classes_right], dim=0)

# pred_masks（必要に応じて）
if instances_left.has("pred_masks") and instances_right.has("pred_masks"):
    masks_left = instances_left.pred_masks
    masks_right = instances_right.pred_masks
    merged_instances.pred_masks = torch.cat([masks_left, masks_right], dim=0)

# 可視化
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("horseshoe_dataset"), scale=1.2)
out = v.draw_instance_predictions(merged_instances)
# 保存・表示
cv2.imwrite(visualized_img_path, out.get_image()[:, :, ::-1])
cv2.imshow("Sample Image", out.get_image()[:, :, ::-1])
cv2.waitKey(10* 1000)
print(f"✅ 可視化画像を {visualized_img_path} に保存しました")

cv2.destroyAllWindows()