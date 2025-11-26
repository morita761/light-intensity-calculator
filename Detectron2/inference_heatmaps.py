from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def extract_region(image, coefficient_top=0.2,
                   coefficient_bottom=0.8,
                   coefficient_left=0.25,
                   coefficient_right=0.66):
    """
    元画像から指定された割合の領域を切り抜きます。
    この関数は緑色抽出を行わず、純粋な画像領域の切り抜きに特化しています。

    Args:
        image (np.array): 入力画像 (BGR形式)。
        coefficient_top (float): 上部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_bottom (float): 下部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_left (float): 左部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_right (float): 右部から切り抜く領域の割合 (0.0 - 1.0)。

    Returns:
        np.array: 指定された領域が切り抜かれた画像。
                  指定領域外は黒（0）になります。
    """
    height, width = image.shape[:2]

    # 切り抜き範囲を計算
    y_start = int(height * coefficient_top)
    y_end = int(height * coefficient_bottom)
    x_start = int(width * coefficient_left)
    x_end = int(width * coefficient_right)

    # 確実に有効なインデックス範囲内に収める
    y_start = max(0, min(y_start, height - 1))
    y_end = max(0, min(y_end, height))
    x_start = max(0, min(x_start, width - 1))
    x_end = max(0, min(x_end, width))

    # 切り抜き領域のマスクを作成 (指定領域内が白、外が黒)
    clip_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    clip_mask[y_start:y_end, x_start:x_end] = 255

    # 元画像にマスクを適用して領域を切り抜く
    clipped_image = cv2.bitwise_and(image, image, mask=clip_mask)

    # 余白をなくして純粋に切り抜かれた矩形領域を返す
    return clipped_image[y_start:y_end, x_start:x_end].copy()


def split_left_right(image):
    """
    入力画像内の緑色領域の垂直方向の中心線X座標を返します。
    このX座標は、後で馬蹄形輪郭を左右に分類するために使用されます。

    Args:
        image (np.array): 緑色領域を含む画像（BGR形式）。

    Returns:
        int: 緑色領域の垂直方向の中心線X座標。
             緑色領域が見つからない場合は、画像全体の中心X座標を返します。
    """
    height, width = image.shape[:2]

    # BGR → HSV に変換して緑色領域を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])  # 緑色のHSV下限値（調整可能）
    upper_green = np.array([80, 255, 255]) # 緑色のHSV上限値（調整可能）
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 緑色領域のバウンディングボックスを計算
    if cv2.countNonZero(green_mask) > 0:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(green_mask))
    else:
        print("警告: 画像内に緑色領域が見つかりませんでした。画像全体の中心で分割します。")
        x, y, w, h = 0, 0, width, height # 画像全体を対象とする

    # 緑色領域の中心X座標を計算
    green_center_x = x + w // 2

    return green_center_x

def create_average_heatmap(masks, original_image, target_size=(100, 100)):
    """
    複数のマスクから平均ヒートマップを生成する関数。
    """
    if not masks:
        print("有効なマスクがありません。ヒートマップは黒くなります。")
        return np.zeros(target_size, dtype=np.uint8)

    aligned_gfp_data = []

    h_img, w_img = original_image.shape[:2]

    for mask in masks:
        # マスクのバウンディングボックスを計算
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            continue

        # バウンディングボックスの座標を画像の範囲内に制限
        y_min = max(0, np.min(y_coords))
        y_max = min(h_img, np.max(y_coords) + 1)
        x_min = max(0, np.min(x_coords))
        x_max = min(w_img, np.max(x_coords) + 1)

        # 座標が有効な範囲か確認
        if y_max <= y_min or x_max <= x_min:
            continue
        
        # マスク領域の輝度データを抽出
        gfp_channel = original_image[:, :, 1]
        gfp_roi = gfp_channel[y_min:y_max, x_min:x_max]

        # マスクを適用
        local_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
        masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=local_mask)

        # リサイズしてサイズを統一
        if masked_gfp_roi.size > 0:
            resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
            aligned_gfp_data.append(resized_gfp)
    
    if not aligned_gfp_data:
        print("リサイズ可能なマスク領域がありませんでした。")
        return np.zeros(target_size, dtype=np.uint8)

    average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0).astype(np.float32)
    final_heatmap = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return final_heatmap


# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "./output/model_0018999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
cfg.MODEL.DEVICE = "cpu"
# cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.2

# --- 推論器 ---
predictor = DefaultPredictor(cfg)

# 複数の入力ファイルパス
# input_paths = ["../data/25cas9P/01.tif", "../data/25cas9P/02.tif", "../data/25cas9P/03.tif"] # 実際のファイルパスに置き換えてください
input_paths = ["./output/004.tif", "./output/6.tif", "./output/003.tif" 
               ] # 実際のファイルパスに置き換えてください


input_path = input_paths[0]

# --- 入力ファイル名やIDを抽出 ---
base_name = os.path.splitext(os.path.basename(input_path))[0]  # '003'

# --- 出力ディレクトリの準備 ---
output_dir = f"./output/test3/{base_name}/"
os.makedirs(output_dir, exist_ok=True)
kmeans_dir = f"{output_dir}/kmeans/"
os.makedirs(kmeans_dir, exist_ok=True)

# === 出力ファイル名の定義 ===
heatmap_path_left = os.path.join(output_dir, f"heatmap_left_{base_name}.png")
heatmap_path_right = os.path.join(output_dir, f"heatmap_right_{base_name}.png")
combined_heatmap_path = os.path.join(output_dir, f"combined_heatmap_{base_name}.png")
visualized_img_path = os.path.join(output_dir, f"result_horseshoe_{base_name}_thresh.png")

# --- 複数の画像を処理 ---
all_masks_left_total = []
all_masks_right_total = []
debug_images = []

for i, input_path in enumerate(input_paths):
    image = cv2.imread(input_path)
    if image is None:
        print(f"警告: 画像ファイル '{input_path}' が読み込めませんでした。スキップします。")
        continue

    # 輪郭検出用のマスク画像を切り抜き
    image_cropped = extract_region(image, coefficient_left=0.23, coefficient_right=0.75, coefficient_bottom=0.92, coefficient_top=0.09)
    green_center_x = split_left_right(image_cropped)

    # --- 推論 ---
    outputs = predictor(image_cropped)

    # インスタンス情報を取得
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    # debug表示用に画像をコピー
    debug_img_manual = image_cropped.copy()
    # 中心線を白で描画
    cv2.line(debug_img_manual, (green_center_x, 0), (green_center_x, debug_img_manual.shape[0]), (255, 255, 255), 1)

    threashold = 50
    # 各輪郭を左右に分類し、GFP輝度データを抽出
    valid_horseshoe_count_left = 0
    valid_horseshoe_count_right = 0

    for j, (mask, box) in enumerate(zip(masks, boxes)):
        # --- 面積計算
        area = np.sum(mask)
        if area < 100 or area > 900:
            continue

        # マスクのバウンディングボックスを計算 (numpyのwhereを使用)
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            continue

        # バウンディングボックスのx座標を計算
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        w = x_max - x_min

        contour_center_x = x_min + w // 2

        # 中心線に被るか、近すぎる輪郭は無視する
        is_overlapping_or_near = (x_min < green_center_x < x_max) or \
                                 (abs(contour_center_x - green_center_x) <= threashold)

        if is_overlapping_or_near:
            color = (0, 255, 255) # BGR形式 (青, 緑, 赤)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
            continue
        elif contour_center_x < green_center_x:
            color = (0, 0, 255)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
            all_masks_left_total.append(mask)
            valid_horseshoe_count_left += 1
        else: # contour_center_x > green_center_x
            color = (255, 0, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
            all_masks_right_total.append(mask)
            valid_horseshoe_count_right += 1

    print(f"画像 {i+1}: 左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")

    # 塗りつぶしたデバッグ画像をdebug_imagesリストに追加
    debug_images.append(cv2.cvtColor(debug_img_manual, cv2.COLOR_BGR2RGB))

# --- ヒートマップの生成と表示 ---
target_size = (100, 100) # ヒートマップのサイズ
heatmap_left = create_average_heatmap(all_masks_left_total, image_cropped, target_size)
heatmap_right = create_average_heatmap(all_masks_right_total, image_cropped, target_size)

num_image_sets = len(debug_images)
num_plots = 3  # 画像、左ヒートマップ、右ヒートマップ
num_rows_for_display = max(num_image_sets, 1)

# plt.figure(figsize=(15, 6 * num_rows_for_display))
fig_width = 15  # 3つのプロットが横に並ぶことを考慮した幅
fig_height = 8 # 画像とヒートマップの2段分の高さを考慮

plt.figure(figsize=(fig_width, fig_height))


# 個別の画像を表示
for i in range(num_image_sets):
    plt.subplot(num_rows_for_display, num_plots, i * num_plots + 1)
    plt.imshow(debug_images[i])
    plt.title(f'Detected Horseshoes {i+1} \n(Cropped)')
    plt.axis('off')

# 左側のヒートマップのプロット
ax = plt.subplot(1, num_plots, 2)
ax.set_aspect('equal')
sns.heatmap(heatmap_left, cmap='viridis', cbar_kws={'label': 'Normalized Average GFP Intensity'}, vmin=0, vmax=255)
plt.title('Average GFP Intensity Heatmap (Left)')
plt.xticks([])
plt.yticks([])

# 右側のヒートマップのプロット
ax = plt.subplot(1, num_plots, 3)
ax.set_aspect('equal')
sns.heatmap(heatmap_right, cmap='viridis', cbar_kws={'label': 'Normalized Average GFP Intensity'}, vmin=0, vmax=255)
plt.title('Average GFP Intensity Heatmap (Right)')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()