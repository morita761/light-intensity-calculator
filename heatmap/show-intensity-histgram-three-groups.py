import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import sys

# 定数設定
EXCLUSION_WIDTH = 50  # 中心線からの無視する距離
DEBUG_FLAG = 0
target_size = (100, 100)

# アノテーションに使用する色のHSV範囲を定義
ANNOTATION_COLORS_HSV = {
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

# --- ユーティリティ関数 ---
def extract_color_mask(mask_image_bgr, color_name):
    """指定された色のHSV範囲に基づいてマスクを抽出する"""
    hsv = cv2.cvtColor(mask_image_bgr, cv2.COLOR_BGR2HSV)
    color_ranges = ANNOTATION_COLORS_HSV[color_name]

    mask1 = cv2.inRange(hsv, color_ranges["lower"], color_ranges["upper"])
    if "lower2" in color_ranges:
        mask2 = cv2.inRange(hsv, color_ranges["lower2"], color_ranges["upper2"])
        final_mask = cv2.bitwise_or(mask1, mask2)
    else:
        final_mask = mask1
    return final_mask

def split_left_right(image):
    """画像全体の中心X座標を分割線とする"""
    height, width = image.shape[:2]
    image_center_x = width // 2
    return image_center_x

# --- メイン処理で使用する画像データ ---
image_pairs_control = [
    {'original': './pics/control/Projections of 241106_Fz_002_second.png',
     'mask': './pics/control/Projections of 241106_Fz_002_second_mask.tif'},
    {'original':'./pics/control/Projections of 241106_Fz_control003.png',
     'mask': './pics/control/Projections of 241106_Fz_control003_mask.tif'},
    {'original':'./pics/control/Projections of 20241024_controlFz001.png',
     'mask': './pics/control/Projections of 20241024_controlFz001_mask.tif'},
    {'original':'./pics/control/Projections of 20241024_controlFz003.png',
     'mask': './pics/control/Projections of 20241024_controlFz003_mask.tif'}
]

# vang RNAi用（画像準備後にパスを変更）
image_pairs_vang_rnai = [
    {'original': './pics/vang-rnai/C1-Projections of 20240929_Fz_GFP_loco_Gal4_UAS_vang_RNAi003.png',
     'mask': './pics/vang-rnai/C1-Projections of 20240929_Fz_GFP_loco_Gal4_UAS_vang_RNAi003_mask.tif'},
    {'original':'./pics/vang-rnai/C1-Projections of 20241024_Fz_GFP.nd2-20241024_Fz_GFP-1.png',
     'mask': './pics/vang-rnai/C1-Projections of 20241024_Fz_GFP.nd2-20241024_Fz_GFP-1_mask.tif'},
    {'original':'./pics/vang-rnai/Projections of C1-20241024_Fz_GFP.nd2-20241024_Fz_GFP_3.png',
     'mask': './pics/vang-rnai/Projections of C1-20241024_Fz_GFP.nd2-20241024_Fz_GFP_3_mask.tif'},
    {'original':'./pics/vang-rnai/Projections of C1-20241024_Fz_GFP.nd2-20241024_Fz_GFP_4.png',
     'mask': './pics/vang-rnai/Projections of C1-20241024_Fz_GFP.nd2-20241024_Fz_GFP_4_mask.tif'}
]

image_pairs_vang_cas9 = [
    {'original': './pics/vang-cas9/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue.png',
     'mask': './pics/vang-cas9/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang-cas9/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita.png',
     'mask': './pics/vang-cas9/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita_mask.tif'},
    {'original':'./pics/vang-cas9/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue.png',
     'mask': './pics/vang-cas9/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang-cas9/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2.png',
     'mask': './pics/vang-cas9/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2_mask.tif'}
]

# 全ての画像ペアを結合（グローバル平均輝度計算のため）
all_image_pairs = image_pairs_control + image_pairs_vang_rnai + image_pairs_vang_cas9

# =========================================================================
# 【STEP 1】 全体の基準平均輝度 (global_mean_intensity) の計算
# =========================================================================
all_raw_gfp_values = []

for file_info in all_image_pairs:
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']

    image_original_for_gfp = cv2.imread(original_file_name)
    image_mask_for_contours = cv2.imread(mask_file_name)

    if image_original_for_gfp is None or image_mask_for_contours is None:
        continue

    gfp_channel = image_original_for_gfp[:, :, 1]

    combined_mask = np.zeros_like(gfp_channel, dtype=np.uint8)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        combined_mask = cv2.bitwise_or(combined_mask, extract_color_mask(image_mask_for_contours, color_name))

    masked_gfp = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_mask)
    non_zero_gfp = masked_gfp[masked_gfp > 0]

    all_raw_gfp_values.extend(non_zero_gfp.flatten())

if all_raw_gfp_values:
    global_mean_intensity = np.mean(all_raw_gfp_values)
else:
    global_mean_intensity = 1.0
print(f"全体の基準平均輝度 (Global Mean): {global_mean_intensity:.2f}")


# =========================================================================
# 【STEP 2】 データ収集処理の関数化
# =========================================================================

def collect_normalized_gfp_data(image_pairs, global_mean_intensity, target_size):
    """
    画像ペアのリストから、正規化・リサイズされた左右のGFP輝度データを収集する。
    """
    all_aligned_gfp_data_left = []
    all_aligned_gfp_data_right = []
    debug_images = []

    for file_info in image_pairs:
        original_file_name = file_info['original']
        mask_file_name = file_info['mask']

        image_original_for_gfp = cv2.imread(original_file_name)
        image_mask_for_contours = cv2.imread(mask_file_name)

        if image_original_for_gfp is None or image_mask_for_contours is None:
            continue

        image_center_x = split_left_right(image_mask_for_contours)
        gfp_channel = image_original_for_gfp[:, :, 1]

        # 現在の画像の平均輝度を計算
        combined_mask_for_current_image = np.zeros_like(gfp_channel, dtype=np.uint8)
        for color_name in ANNOTATION_COLORS_HSV.keys():
            combined_mask_for_current_image = cv2.bitwise_or(combined_mask_for_current_image, extract_color_mask(image_mask_for_contours, color_name))

        masked_gfp_for_mean = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_mask_for_current_image)
        current_image_gfp_values = masked_gfp_for_mean[masked_gfp_for_mean > 0]

        if current_image_gfp_values.size > 0:
            current_image_mean_intensity = np.mean(current_image_gfp_values)
        else:
            current_image_mean_intensity = global_mean_intensity

        scaling_factor = global_mean_intensity / current_image_mean_intensity
        print(f"   - 画像の平均輝度: {current_image_mean_intensity:.2f}, スケーリングファクター: {scaling_factor:.3f}")

        debug_img = image_original_for_gfp.copy()
        cv2.line(debug_img, (image_center_x, 0), (image_center_x, debug_img.shape[0]), (255, 255, 255), 1)

        all_horseshoe_masks = []
        for color_name in ANNOTATION_COLORS_HSV.keys():
            horseshoe_mask = extract_color_mask(image_mask_for_contours, color_name)
            all_horseshoe_masks.append(horseshoe_mask)

        for single_horseshoe_mask in all_horseshoe_masks:
            contours, _ = cv2.findContours(single_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            for contour in contours:
                if cv2.contourArea(contour) < 20:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                contour_center_x = x + w // 2

                is_overlapping = (x < image_center_x < x + w)

                if is_overlapping or abs(contour_center_x - image_center_x) <= EXCLUSION_WIDTH:
                    continue
                elif contour_center_x < image_center_x:
                    current_gfp_data_list = all_aligned_gfp_data_left
                    color = (0, 0, 255)
                else:
                    current_gfp_data_list = all_aligned_gfp_data_right
                    color = (255, 0, 0)

                temp_mask = np.zeros((h, w), dtype=np.uint8)
                rel_contour = contour - np.array([x, y])
                cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)

                contour_mask_local = temp_mask.copy()

                final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if final_contours:
                    final_absolute_contour = final_contours[0] + np.array([x, y])
                    cv2.drawContours(debug_img, [final_absolute_contour], -1, color, -1)

                gfp_roi = gfp_channel[y:y+h, x:x+w]
                masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)

                resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)

                resized_gfp_float = resized_gfp.astype(np.float64)
                resized_gfp_normalized_float = resized_gfp_float * scaling_factor

                current_gfp_data_list.append(resized_gfp_normalized_float)

        debug_images.append(debug_img)
    return all_aligned_gfp_data_left, all_aligned_gfp_data_right, debug_images


# =========================================================================
# 【STEP 3】 3系統のデータ収集
# =========================================================================

print("\n--- Controlデータ収集開始 ---")
control_left_data, control_right_data, debug_images_control = collect_normalized_gfp_data(
    image_pairs_control, global_mean_intensity, target_size
)

print("\n--- vang RNAiデータ収集開始 ---")
vang_rnai_left_data, vang_rnai_right_data, debug_images_vang_rnai = collect_normalized_gfp_data(
    image_pairs_vang_rnai, global_mean_intensity, target_size
)

print("\n--- vang Cas9データ収集開始 ---")
vang_cas9_left_data, vang_cas9_right_data, debug_images_vang_cas9 = collect_normalized_gfp_data(
    image_pairs_vang_cas9, global_mean_intensity, target_size
)


# =========================================================================
# 【STEP 4】 左右それぞれの平均ヒートマップの計算
# =========================================================================

def calculate_average_heatmap(data_list, label):
    if data_list:
        avg_heatmap = np.mean(np.array(data_list), axis=0)
        print(f"{label} ({len(data_list)}個のデータ) の平均ヒートマップを計算しました。")
        return avg_heatmap
    else:
        print(f"{label} のデータが不足しているため、平均ヒートマップはゼロで初期化されます。")
        return np.zeros(target_size, dtype=np.float64)

def normalize_average_heatmap(data_list, label, max_val):
    avg_heatmap_n = (data_list / max_val * 255).astype(np.uint8)
    return avg_heatmap_n

def get_max_of_all(*arrays):
    max_vals = [np.max(arr) for arr in arrays]
    return max(max_vals)

# 各系統の平均ヒートマップを計算
avg_control_left = calculate_average_heatmap(control_left_data, "Control Left")
avg_control_right = calculate_average_heatmap(control_right_data, "Control Right")

avg_vang_rnai_left = calculate_average_heatmap(vang_rnai_left_data, "vang RNAi Left")
avg_vang_rnai_right = calculate_average_heatmap(vang_rnai_right_data, "vang RNAi Right")

avg_vang_cas9_left = calculate_average_heatmap(vang_cas9_left_data, "vang Cas9 Left")
avg_vang_cas9_right = calculate_average_heatmap(vang_cas9_right_data, "vang Cas9 Right")

# 全体の最大値を取得（カラーバーを統一するため）
global_max = get_max_of_all(
    avg_control_left, avg_control_right,
    avg_vang_rnai_left, avg_vang_rnai_right,
    avg_vang_cas9_left, avg_vang_cas9_right
)
print(f"全体の最大輝度値: {global_max:.2f}")

# ヒートマップを正規化
avg_control_left_norm = normalize_average_heatmap(avg_control_left, "Control Left", global_max)
avg_control_right_norm = normalize_average_heatmap(avg_control_right, "Control Right", global_max)
avg_vang_rnai_left_norm = normalize_average_heatmap(avg_vang_rnai_left, "vang RNAi Left", global_max)
avg_vang_rnai_right_norm = normalize_average_heatmap(avg_vang_rnai_right, "vang RNAi Right", global_max)
avg_vang_cas9_left_norm = normalize_average_heatmap(avg_vang_cas9_left, "vang Cas9 Left", global_max)
avg_vang_cas9_right_norm = normalize_average_heatmap(avg_vang_cas9_right, "vang Cas9 Right", global_max)


# =========================================================================
# 【STEP 5】 各系統ごとに1枚の画像を出力
# =========================================================================

def plot_single_genotype(debug_images, left_heatmap, right_heatmap, genotype_name, output_filename):
    """1系統分のヒートマップを1枚の画像として出力"""
    num_plots = 3
    rows_inner = 2
    cols_inner = 2
    num_debug_images = len(debug_images)

    fig = plt.figure(figsize=(num_plots * 4, 4))

    if num_debug_images >= 4:
        # 外側の GridSpec を定義 (図全体を1行 num_plots列に分割)
        gs_outer = gridspec.GridSpec(1, num_plots, figure=fig)

        # 1区画目 (左端) の中に、内側の GridSpec (2行2列) をネストして定義
        gs_inner = gridspec.GridSpecFromSubplotSpec(rows_inner, cols_inner,
                                                    subplot_spec=gs_outer[0, 0],
                                                    wspace=0.1, hspace=0.1)

        # 2x2 の各セルにデバッグ画像をプロット
        for i in range(num_debug_images):
            if i >= 4:
                break
            if i < rows_inner * cols_inner:
                ax = fig.add_subplot(gs_inner[i])
                ax.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
                ax.set_title(f'Annotated {i+1}', fontsize=10)
                ax.axis('off')

        # 左側のヒートマップのプロット
        ax_left = fig.add_subplot(gs_outer[0, 1])
        im_left = ax_left.imshow(left_heatmap, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(im_left, ax=ax_left, label='Normalized Average GFP Intensity')
        ax_left.set_title(f'{genotype_name}\nAverage GFP Intensity Heatmap \n(Ventral)')
        ax_left.axis('off')

        # 右側のヒートマップのプロット
        ax_right = fig.add_subplot(gs_outer[0, 2])
        im_right = ax_right.imshow(right_heatmap, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(im_right, ax=ax_right, label='Normalized Average GFP Intensity')
        ax_right.set_title(f'{genotype_name}\nAverage GFP Intensity Heatmap \n(Dorsal)')
        ax_right.axis('off')
    else:
        # Debug画像が4枚未満の場合
        plot_index = 1
        for i in range(num_debug_images):
            plt.subplot(1, num_plots, 1)
            plt.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
            plt.title(f'Annotated Image {i+1}')
            plt.axis('off')

        # 左側のヒートマップのプロット
        plt.subplot(1, num_plots, 2)
        plt.imshow(left_heatmap, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(label='Normalized Average GFP Intensity')
        plt.title(f'{genotype_name}\nAverage GFP Intensity Heatmap \n(Ventral)')
        plt.axis('off')

        # 右側のヒートマップのプロット
        plt.subplot(1, num_plots, 3)
        plt.imshow(right_heatmap, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(label='Normalized Average GFP Intensity')
        plt.title(f'{genotype_name}\nAverage GFP Intensity Heatmap \n(Dorsal)')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"{output_filename} を保存しました。")

# 各系統ごとに画像を出力
plot_single_genotype(debug_images_control, avg_control_left_norm, avg_control_right_norm,
                     'Control', 'heatmap_control.png')

plot_single_genotype(debug_images_vang_rnai, avg_vang_rnai_left_norm, avg_vang_rnai_right_norm,
                     'vang RNAi', 'heatmap_vang_rnai.png')

plot_single_genotype(debug_images_vang_cas9, avg_vang_cas9_left_norm, avg_vang_cas9_right_norm,
                     'vang Cas9', 'heatmap_vang_cas9.png')


# =========================================================================
# 【STEP 6】 3系統のデバッグ用ヒストグラム
# =========================================================================

def plot_histogram_comparison(data_lists, labels, colors, title):
    """複数のデータセットのヒストグラムを重ねて表示"""
    plt.figure(figsize=(10, 6))

    max_val = 0
    for data_list in data_lists:
        if data_list:
            flat_data = np.concatenate([arr.flatten() for arr in data_list])
            max_val = max(max_val, np.max(flat_data))

    for data_list, label, color in zip(data_lists, labels, colors):
        if data_list:
            flat_data = np.concatenate([arr.flatten() for arr in data_list])
            plt.hist(flat_data, bins=50, range=(0, max_val), color=color, alpha=0.5, label=label)

    plt.title(title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

# Left (Ventral) のヒストグラム比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left側
plt.subplot(1, 2, 1)
left_data_lists = [control_left_data, vang_rnai_left_data, vang_cas9_left_data]
labels = ['Control', 'vang RNAi', 'vang Cas9']
colors = ['blue', 'green', 'red']

max_val_left = 0
for data_list in left_data_lists:
    if data_list:
        flat_data = np.concatenate([arr.flatten() for arr in data_list])
        max_val_left = max(max_val_left, np.max(flat_data))

for data_list, label, color in zip(left_data_lists, labels, colors):
    if data_list:
        flat_data = np.concatenate([arr.flatten() for arr in data_list])
        plt.hist(flat_data, bins=50, range=(0, max_val_left), color=color, alpha=0.5, label=label)

plt.title('Normalized GFP Intensity Distribution (Ventral/Left)')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.5)

# Right側
plt.subplot(1, 2, 2)
right_data_lists = [control_right_data, vang_rnai_right_data, vang_cas9_right_data]

max_val_right = 0
for data_list in right_data_lists:
    if data_list:
        flat_data = np.concatenate([arr.flatten() for arr in data_list])
        max_val_right = max(max_val_right, np.max(flat_data))

for data_list, label, color in zip(right_data_lists, labels, colors):
    if data_list:
        flat_data = np.concatenate([arr.flatten() for arr in data_list])
        plt.hist(flat_data, bins=50, range=(0, max_val_right), color=color, alpha=0.5, label=label)

plt.title('Normalized GFP Intensity Distribution (Dorsal/Right)')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('histogram_three_groups.png', dpi=300, bbox_inches='tight')
plt.show()

cv2.waitKey(1)
cv2.destroyAllWindows()
