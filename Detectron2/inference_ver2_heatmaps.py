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
import sys
import argparse

from func.cli_errors import require_file, run_main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from figure_style import plot_vd_heatmap_figure  # noqa: E402

# --- Detectron2 Configuration ---
def setup_predictor(model_path, score_threshold=0.25):
    """
    Sets up and returns a Detectron2 predictor with a custom configuration.
    """
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.2
    return DefaultPredictor(cfg)

# --- Helper Functions (as in original code) ---
def extract_region(image, coefficient_top=0.2,
                   coefficient_bottom=0.8,
                   coefficient_left=0.25,
                   coefficient_right=0.66):
    """
    元画像から指定された割合の領域を切り抜きます。
    この関数は緑色抽出を行わず、純粋な画像領域の切り抜きに特化しています。
    """
    height, width = image.shape[:2]
    y_start = int(height * coefficient_top)
    y_end = int(height * coefficient_bottom)
    x_start = int(width * coefficient_left)
    x_end = int(width * coefficient_right)
    y_start = max(0, min(y_start, height - 1))
    y_end = max(0, min(y_end, height))
    x_start = max(0, min(x_start, width - 1))
    x_end = max(0, min(x_end, width))
    clip_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    clip_mask[y_start:y_end, x_start:x_end] = 255
    clipped_image = cv2.bitwise_and(image, image, mask=clip_mask)
    return clipped_image[y_start:y_end, x_start:x_end].copy()

def split_left_right(image):
    """
    入力画像内の緑色領域の垂直方向の中心線X座標を返します。
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    if cv2.countNonZero(green_mask) > 0:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(green_mask))
    else:
        print("警告: 画像内に緑色領域が見つかりませんでした。画像全体の中心で分割します。")
        x, y, w, h = 0, 0, width, height
    green_center_x = x + w // 2
    return green_center_x

def create_average_heatmap(masks, original_image, target_size=(100, 100)):
    """
    複数のマスクから平均ヒートマップを生成する関数。

    戻り値は正規化前の平均GFP輝度（float, 元画像と同じ0-255スケール）。
    左右（V/D）のヒートマップをそれぞれ独立に0-255へmin-max正規化すると、
    暗い側でも「その中で一番明るいピクセル」が255になり、V/D間の絶対的な
    明るさの違いが図から読み取れなくなるため、ここでは正規化しない。
    表示時のスケーリングは figure_style.plot_vd_heatmap_figure() が
    左右で共有するvmin/vmaxを使って行う。
    """
    if not masks:
        print("有効なマスクがありません。ヒートマップは黒くなります。")
        return np.zeros(target_size, dtype=np.float32)
    aligned_gfp_data = []
    h_img, w_img = original_image.shape[:2]
    for mask in masks:
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            continue
        y_min, y_max = np.min(y_coords), np.max(y_coords) + 1
        x_min, x_max = np.min(x_coords), np.max(x_coords) + 1
        if y_max <= y_min or x_max <= x_min:
            continue
        gfp_channel = original_image[:, :, 1]
        gfp_roi = gfp_channel[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
        masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=local_mask)
        if masked_gfp_roi.size > 0:
            resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
            aligned_gfp_data.append(resized_gfp)
    if not aligned_gfp_data:
        print("リサイズ可能なマスク領域がありませんでした。")
        return np.zeros(target_size, dtype=np.float32)
    average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0).astype(np.float32)
    return average_heatmap

# --- Main Processing Logic ---
def process_images(input_paths, predictor, class_names):
    """
    Processes a list of images, filters for the 'horseshoe' class,
    and generates heatmaps and visualizations.
    """
    all_masks_left_total = []
    all_masks_right_total = []
    debug_images = []
    image_cropped = None

    # ["horseshoe", "negative_green_region"]
    horseshoe_class_id = class_names.index('horseshoe')

    for i, input_path in enumerate(input_paths):
        image = cv2.imread(input_path)
        if image is None:
            print(f"警告: 画像ファイル '{input_path}' が読み込めませんでした。スキップします。")
            continue

        image_cropped = extract_region(image, coefficient_left=0.23, coefficient_right=0.75, coefficient_bottom=0.92, coefficient_top=0.09)
        green_center_x = split_left_right(image_cropped)
        outputs = predictor(image_cropped)
        instances = outputs["instances"].to("cpu")
        
        # --- ここが修正点 ---
        # 予測クラスが'horseshoe'のものだけを抽出
        horseshoe_instances = instances[instances.pred_classes == horseshoe_class_id]
        if not horseshoe_instances:
            print(f"警告: 画像 '{input_path}' に 'horseshoe' クラスのオブジェクトが見つかりませんでした。")
            continue
            
        masks = horseshoe_instances.pred_masks.numpy()
        boxes = horseshoe_instances.pred_boxes.tensor.numpy()
        
        debug_img_manual = image_cropped.copy()
        cv2.line(debug_img_manual, (green_center_x, 0), (green_center_x, debug_img_manual.shape[0]), (255, 255, 255), 1)

        valid_horseshoe_count_left = 0
        valid_horseshoe_count_right = 0
        
        threashold = 50

        for j, mask in enumerate(masks):
            area = np.sum(mask)
            if not (100 <= area <= 5000):
                continue
            
            y_coords, x_coords = np.where(mask)
            if len(x_coords) == 0:
                continue

            x_min, x_max = np.min(x_coords), np.max(x_coords)
            w = x_max - x_min
            contour_center_x = x_min + w // 2
            
            is_overlapping_or_near = (x_min < green_center_x < x_max) or \
                                     (abs(contour_center_x - green_center_x) <= threashold)
            
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            if is_overlapping_or_near:
                color = (0, 255, 255)  # Yellow for ignored
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
            elif contour_center_x < green_center_x:
                color = (0, 0, 255)  # Red for left
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
                all_masks_left_total.append(mask)
                valid_horseshoe_count_left += 1
            else:
                color = (255, 0, 0)  # Blue for right
                cv2.drawContours(debug_img_manual, contours, -1, color, cv2.FILLED)
                all_masks_right_total.append(mask)
                valid_horseshoe_count_right += 1

        print(f"画像 {i+1}: 左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")
        debug_images.append(cv2.cvtColor(debug_img_manual, cv2.COLOR_BGR2RGB))
        
    return all_masks_left_total, all_masks_right_total, debug_images, image_cropped

def plot_debug_grid(debug_images):
    """
    入力画像ごとの検出結果（クロップ+輪郭色分け）を並べた診断用の図。
    論文用ではなく、検出が正しく左右分類できているかの確認用。
    """
    if not debug_images:
        return
    num_image_sets = len(debug_images)
    cols = min(3, num_image_sets)
    rows = (num_image_sets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_image_sets:
            ax.imshow(debug_images[i])
            ax.set_title(f'Detected horseshoes {i + 1} (cropped)', fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_results(heatmap_left, heatmap_right, n_left, n_right):
    """
    V/D平均輝度ヒートマップの比較図（論文品質、左右で共有カラースケール）。
    """
    fig, axes = plot_vd_heatmap_figure(
        heatmap_left, heatmap_right,
        n_v=n_left, n_d=n_right,
        v_label="Ventral (V)", d_label="Dorsal (D)",
        colorbar_label="Mean GFP intensity (a.u.)",
    )
    plt.show()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Detectron2による馬蹄形検出 + V-D分割ヒートマップ生成（改良版）"
    )
    parser.add_argument(
        "--model-path", default="./output/model_final.pth", help="学習済みモデル重み(.pth)のパス"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["./output/004.tif", "./output/6.tif", "./output/003.tif"],
        help="推論対象の入力画像パス（複数指定可）",
    )
    parser.add_argument("--score-threshold", type=float, default=0.3, help="検出の信頼度閾値")
    args = parser.parse_args()

    require_file(args.model_path, "学習済みモデル(--model-path)")
    existing_images = [p for p in args.images if os.path.isfile(p)]
    if not existing_images:
        raise FileNotFoundError(
            f"--images で指定された画像が1枚も見つかりません: {args.images}"
        )
    missing_images = set(args.images) - set(existing_images)
    for p in missing_images:
        print(f"警告: 画像ファイル '{p}' が見つかりません。スキップします。")

    predictor = setup_predictor(args.model_path, score_threshold=args.score_threshold)

    # ここでクラス名を定義。モデルの学習時のクラス順序と一致させる必要があります。
    # 0:'background', 1:'horseshoe'
    class_names = ["horseshoe", "negative_green_region"]
    MetadataCatalog.get("my_dataset_train").set(thing_classes=class_names)

    all_masks_left, all_masks_right, debug_images, image_cropped = process_images(
        existing_images, predictor, class_names
    )
    if image_cropped is None:
        raise ValueError(
            "指定された画像から馬蹄形を検出できませんでした（全画像で読み込み/検出に失敗）。"
        )

    target_size = (100, 100)
    heatmap_left = create_average_heatmap(all_masks_left, image_cropped, target_size)
    heatmap_right = create_average_heatmap(all_masks_right, image_cropped, target_size)

    plot_debug_grid(debug_images)
    plot_results(heatmap_left, heatmap_right, len(all_masks_left), len(all_masks_right))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main(main)