import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.gridspec as gridspec

# 除外する中心線の幅を定義
EXCLUSION_WIDTH = 50

# デバックフラグ (on: 1, off: 0)
DEBUG_FLAG = 0

# アノテーションに使用する色のHSV範囲を定義
ANNOTATION_COLORS_HSV = {
    # "red": {
    #     "lower": np.array([0, 100, 100]),
    #     "upper": np.array([10, 255, 255]),
    #     "lower2": np.array([170, 100, 100]),
    #     "upper2": np.array([180, 255, 255])
    # },
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

def split_left_right(image):
    """画像全体の中心を左右分割線とする"""
    height, width = image.shape[:2]
    image_center_x = width // 2 
    print(f"分割中心X座標: {image_center_x} (画像の中心)")
    return image_center_x

# --- メイン処理で使用するヘルパー関数 ---

def process_image_set(file_info, all_raw_gfp_values, image_center_x, scaling_factor, target_size, data_list_left, data_list_right, is_rfp_mask=False):
    """
    単一の画像セット（maskまたはrfp_mask）からGFP輝度データを抽出・正規化し、データリストに追加する。
    is_rfp_maskがTrueの場合、rfp_maskファイルを使用し、rfp_mask用のデータリストに格納する。
    """
    # 1. 画像の読み込み
    original_file_name = file_info['original']
    mask_file_name = file_info['rfp_mask'] if is_rfp_mask else file_info['mask']
    
    image_original_for_gfp = cv2.imread(original_file_name)
    image_mask_for_contours = cv2.imread(mask_file_name)
    
    if image_original_for_gfp is None or image_mask_for_contours is None:
        print(f"画像を読み込めませんでした: {original_file_name} または {mask_file_name}。スキップします。")
        return 0, 0
        
    gfp_channel = image_original_for_gfp[:, :, 1] # GFPチャネル（緑）
    
    valid_horseshoe_count_left = 0
    valid_horseshoe_count_right = 0
    
    all_horseshoe_masks = []
    # extract_color_maskを使って、馬蹄形輪郭を抽出 (全ての定義済み色を結合)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        horseshoe_mask = extract_color_mask(image_mask_for_contours, color_name)
        all_horseshoe_masks.append(horseshoe_mask)

    # 各輪郭を左右に分類し、GFP輝度データを抽出
    for single_horseshoe_mask in all_horseshoe_masks:
        # 個々のマスクから輪郭を検出
        contours, _ = cv2.findContours(single_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 20:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_x = x + w // 2
            
            threashold = 10 # 中心線からの無視する距離
            
            is_overlapping = (x < image_center_x < x + w)
            if is_overlapping:
                continue 
            elif contour_center_x + threashold < image_center_x:
                current_gfp_data_list = data_list_left
                valid_horseshoe_count_left += 1
            elif contour_center_x - threashold > image_center_x:
                current_gfp_data_list = data_list_right
                valid_horseshoe_count_right += 1
            else:
                continue

            # 膨張処理（元のコードのロジックを維持）
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            rel_contour = contour - np.array([x, y])
            cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)
            
            kernel = np.ones((3, 3), np.uint8)
            iterations = 0
            contour_mask_local = temp_mask.copy()
            
            while True:
                total_pixels = w * h
                white_pixels = cv2.countNonZero(contour_mask_local)
                occupancy_rate = white_pixels / total_pixels if total_pixels > 0 else 0
                
                if occupancy_rate > 0.60 or iterations >= 4:
                    break

                contour_mask_local = cv2.dilate(contour_mask_local, kernel, iterations=1)
                iterations += 1

            if iterations > 2: # 膨張が3回以上ならスキップ
                continue
                
            # GFPチャネル（オリジナル画像から抽出）からROIを切り出し
            gfp_roi = gfp_channel[y:y+h, x:x+w]

            # 切り出したGFP輝度データに、作成したローカルマスクを適用
            masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)

            # リサイズ (target_sizeに統一)
            resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
            
            # 輝度正規化の適用
            resized_gfp_float = resized_gfp.astype(np.float64) 
            resized_gfp_normalized_float = resized_gfp_float * scaling_factor
            
            # 結果を uint8 に戻し、クリッピング
            resized_gfp_final = np.clip(resized_gfp_normalized_float, 0, 255).astype(np.uint8)
            
            current_gfp_data_list.append(resized_gfp_final)
            
    return valid_horseshoe_count_left, valid_horseshoe_count_right


# --- メイン処理 ---
# image_pairs = [
#     {'original': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/C1-Projections of 20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063.png', 
#      'mask': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_normal.tif',
#      'rfp': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged.png',
#      'rfp_mask': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_rfp.tif'},
#     {'original':'./pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/C1-Projections of 20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063.png', 
#      'mask': './pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_normal.tif',
#      'rfp': './pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged.png',
#      'rfp_mask': './pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_rfp.tif'},
# ]

image_pairs = [
    {'original': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/C1-Projections of 20250423_24%APF_Fz-GFP_RFP_ue.png', 
     'mask': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/Merged_normal.tif',
     'rfp': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/Merged.png',
     'rfp_mask': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/Merged_rfp.tif'},     
    {'original':'./pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/C1-Projections of 20250428_24%APF_Fz-GFP_sensRFP.png', 
     'mask': './pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/Merged_rfp.tif',
     'rfp': './pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/Merged.png',
     'rfp_mask': './pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/Merged_rfp.tif'},     
]

target_size = (100, 100) 

# =========================================================================
# 【STEP 1】 全体の基準平均輝度 (global_mean_intensity) の計算 (Mask + RFP Mask)
# =========================================================================
all_raw_gfp_values = [] 

for file_info in image_pairs:
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']
    rfp_mask_file_name = file_info['rfp_mask']
    
    image_original_for_gfp = cv2.imread(original_file_name)
    image_mask_for_contours = cv2.imread(mask_file_name)
    image_rfp_mask_for_contours = cv2.imread(rfp_mask_file_name) # RFPマスクを読み込み
    
    if image_original_for_gfp is None or image_mask_for_contours is None or image_rfp_mask_for_contours is None:
        continue
        
    gfp_channel = image_original_for_gfp[:, :, 1]
    
    # 全ての色のマスク（normal mask）を結合
    combined_normal_mask = np.zeros_like(gfp_channel, dtype=np.uint8)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        combined_normal_mask = cv2.bitwise_or(combined_normal_mask, extract_color_mask(image_mask_for_contours, color_name))
        
    # 全ての色のマスク（rfp mask）を結合
    combined_rfp_mask = np.zeros_like(gfp_channel, dtype=np.uint8)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        combined_rfp_mask = cv2.bitwise_or(combined_rfp_mask, extract_color_mask(image_rfp_mask_for_contours, color_name))

    # 結合マスク内のGFP輝度値を取得（normal mask）
    masked_gfp_normal = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_normal_mask)
    non_zero_gfp_normal = masked_gfp_normal[masked_gfp_normal > 0]
    all_raw_gfp_values.extend(non_zero_gfp_normal.flatten())

    # 結合マスク内のGFP輝度値を取得（rfp mask）
    masked_gfp_rfp = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_rfp_mask)
    non_zero_gfp_rfp = masked_gfp_rfp[masked_gfp_rfp > 0]
    all_raw_gfp_values.extend(non_zero_gfp_rfp.flatten()) # RFPマスクのデータも全体平均に含める
    
# 全体の基準平均輝度 (Global Mean Intensity) を計算
if all_raw_gfp_values:
    global_mean_intensity = np.mean(all_raw_gfp_values) 
else:
    global_mean_intensity = 1.0 # データがない場合は0割を防ぐ
print(f"✅ 全体の基準平均輝度 (Global Mean): {global_mean_intensity:.2f}")

# =========================================================================
# 【STEP 2-A】 個々の馬蹄形データ（Normal Mask）の抽出と輝度正規化
# =========================================================================
all_aligned_gfp_data_left = []
all_aligned_gfp_data_right = []
all_aligned_rfp_gfp_data_left = [] # RFPマスク用のデータリスト
all_aligned_rfp_gfp_data_right = [] # RFPマスク用のデータリスト
debug_images = []

for index, file_info in enumerate(image_pairs):
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']
    rfp_file_name = file_info['rfp'] # アノテーション描画用
    
    # 1. 画像の読み込みと分割中心の取得
    image_original_for_gfp = cv2.imread(original_file_name)
    image_mask_for_contours = cv2.imread(mask_file_name)
    image_original_for_rfp = cv2.imread(rfp_file_name) # RFP画像 (描画ベース)

    if image_original_for_gfp is None or image_mask_for_contours is None or image_original_for_rfp is None:
        print(f"画像を読み込めませんでした: {original_file_name}, {mask_file_name} または {rfp_file_name}。スキップします。")
        continue

    # 左右分割の中心X座標を取得
    image_center_x = split_left_right(image_mask_for_contours)
    gfp_channel = image_original_for_gfp[:, :, 1] # GFPチャネル（緑）
    
    # 2. 現在の画像の平均輝度を計算し、スケーリングファクターを決定
    combined_mask_for_current_image = np.zeros_like(gfp_channel, dtype=np.uint8)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        combined_mask_for_current_image = cv2.bitwise_or(combined_mask_for_current_image, extract_color_mask(image_mask_for_contours, color_name))
        
    masked_gfp_for_mean = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_mask_for_current_image)
    current_image_gfp_values = masked_gfp_for_mean[masked_gfp_for_mean > 0]

    if current_image_gfp_values.size > 0:
        current_image_mean_intensity = np.mean(current_image_gfp_values)
    else:
        current_image_mean_intensity = global_mean_intensity # 0割防止
        
    scaling_factor = global_mean_intensity / current_image_mean_intensity
    print(f"   - 画像の平均輝度: {current_image_mean_intensity:.2f}, スケーリングファクター: {scaling_factor:.3f}")
    
    # 3. アノテーション画像の準備（RFP画像をベースとする）
    debug_img = image_original_for_rfp.copy() # RFP画像をベースとしてコピー
    cv2.line(debug_img, (image_center_x, 0), (image_center_x, debug_img.shape[0]), (255, 255, 255), 1)

    # 4. Normal Maskからのデータ抽出とアノテーション描画
    all_horseshoe_masks_normal = []
    for color_name in ANNOTATION_COLORS_HSV.keys():
        all_horseshoe_masks_normal.append(extract_color_mask(image_mask_for_contours, color_name))

    valid_horseshoe_count_left, valid_horseshoe_count_right = 0, 0

    for single_horseshoe_mask in all_horseshoe_masks_normal:
        contours, _ = cv2.findContours(single_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_x = x + w // 2
            
            threashold = 10 
            is_overlapping = (x < image_center_x < x + w)
            if is_overlapping:
                continue 
            elif contour_center_x + threashold < image_center_x:
                current_gfp_data_list = all_aligned_gfp_data_left
                color = (255, 0, 255) # マゼンタ (BGR) for debug/annotation (Left)
                valid_horseshoe_count_left += 1
            elif contour_center_x - threashold > image_center_x:
                current_gfp_data_list = all_aligned_gfp_data_right
                color = (255, 0, 0) # 青 (BGR) for debug/annotation (Right)
                valid_horseshoe_count_right += 1
            else:
                continue

            # 膨張処理（元のコードのロジックを維持）
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            rel_contour = contour - np.array([x, y])
            cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)
            
            kernel = np.ones((3, 3), np.uint8)
            iterations = 0
            contour_mask_local = temp_mask.copy()
            while True:
                total_pixels = w * h
                white_pixels = cv2.countNonZero(contour_mask_local)
                occupancy_rate = white_pixels / total_pixels if total_pixels > 0 else 0
                if occupancy_rate > 0.60 or iterations >= 4:
                    break
                contour_mask_local = cv2.dilate(contour_mask_local, kernel, iterations=1)
                iterations += 1

            if iterations > 2: 
                continue
                
            # アノテーションの描画 (RFP画像をベースに半透明で)
            final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                final_absolute_contour = final_contours[0] + np.array([x, y])
                overlay = debug_img.copy()
                alpha = 0.4 # 半透明度
                cv2.drawContours(overlay, [final_absolute_contour], -1, color, -1)
                debug_img = cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0)

            # GFP輝度データ抽出・正規化
            gfp_roi = gfp_channel[y:y+h, x:x+w]
            masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)
            resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
            resized_gfp_float = resized_gfp.astype(np.float64) 
            resized_gfp_normalized_float = resized_gfp_float * scaling_factor
            resized_gfp_final = np.clip(resized_gfp_normalized_float, 0, 255).astype(np.uint8)
            
            current_gfp_data_list.append(resized_gfp_final)
            
    cv2.imshow(f"Annotated Image (Normal Mask): {os.path.basename(original_file_name)}", debug_img)
    cv2.waitKey(1) 
    debug_images.append(debug_img) # 統合プロット用に格納

    print(f"'{original_file_name}' から左側の有効な馬蹄形(Normal)が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形(Normal)が {valid_horseshoe_count_right} 個検出されました。")

    # =========================================================================
    # 【STEP 2-B】 個々の馬蹄形データ（RFP Mask）の抽出と輝度正規化
    # =========================================================================
    rfp_valid_left, rfp_valid_right = process_image_set(
        file_info, 
        all_raw_gfp_values, 
        image_center_x, 
        scaling_factor, 
        target_size, 
        all_aligned_rfp_gfp_data_left, 
        all_aligned_rfp_gfp_data_right, 
        is_rfp_mask=True
    )
    print(f"'{original_file_name}' から左側の有効な馬蹄形(RFP Mask)が {rfp_valid_left} 個、右側の有効な馬蹄形(RFP Mask)が {rfp_valid_right} 個検出されました。")


# =========================================================================
# 【STEP 3】 Heatmapの計算と、正規化最大値の決定
# =========================================================================

# --- Normal Mask Heatmapの計算 ---
average_heatmap_left = np.mean(np.array(all_aligned_gfp_data_left), axis=0) if all_aligned_gfp_data_left else np.zeros(target_size, dtype=np.float64)
average_heatmap_right = np.mean(np.array(all_aligned_gfp_data_right), axis=0) if all_aligned_gfp_data_right else np.zeros(target_size, dtype=np.float64)

# --- RFP Mask Heatmapの計算 ---
rfp_average_heatmap_left = np.mean(np.array(all_aligned_rfp_gfp_data_left), axis=0) if all_aligned_rfp_gfp_data_left else np.zeros(target_size, dtype=np.float64)
rfp_average_heatmap_right = np.mean(np.array(all_aligned_rfp_gfp_data_right), axis=0) if all_aligned_rfp_gfp_data_right else np.zeros(target_size, dtype=np.float64)

# --- Normal Maskの正規化 (最大値の決定用) ---
# Normal Maskの左右のヒートマップ全体で最大の輝度値を取得
max_gfp_intensity_normal = 0.0
if average_heatmap_left.size > 0 and np.max(average_heatmap_left) > max_gfp_intensity_normal:
    max_gfp_intensity_normal = np.max(average_heatmap_left)
if average_heatmap_right.size > 0 and np.max(average_heatmap_right) > max_gfp_intensity_normal:
    max_gfp_intensity_normal = np.max(average_heatmap_right)

if max_gfp_intensity_normal == 0.0:
    max_gfp_intensity_normal = 1.0 # 0除算を防ぐ

# Normal Maskのヒートマップの最終的な正規化
average_heatmap_left_normalized = np.clip((average_heatmap_left / max_gfp_intensity_normal) * 255.0, 0, 255).astype(np.uint8)
average_heatmap_right_normalized = np.clip((average_heatmap_right / max_gfp_intensity_normal) * 255.0, 0, 255).astype(np.uint8)

print(f"✅ Normal Mask ヒートマップの最大値: {max_gfp_intensity_normal:.2f} (この値がRFPヒートマップの正規化にも使われます)")

# --- RFP Maskの正規化 (Normal Maskの最大値を考慮) ---
# D'' = D' * (255 / I_normal_max)
rfp_average_heatmap_left_normalized = np.clip((rfp_average_heatmap_left / max_gfp_intensity_normal) * 255.0, 0, 255).astype(np.uint8)
rfp_average_heatmap_right_normalized = np.clip((rfp_average_heatmap_right / max_gfp_intensity_normal) * 255.0, 0, 255).astype(np.uint8)


# =========================================================================
# 【STEP 4】 ヒートマップの表示（Normal + RFP）
# =========================================================================
if DEBUG_FLAG == 1:
    # Normal Maskのヒートマップ個別出力
    # ... (元のコードのまま、ここでは省略)

    # RFP Maskのヒートマップ個別出力
    if all_aligned_rfp_gfp_data_left:
        rfp_left_heatmap_colored = cv2.applyColorMap(rfp_average_heatmap_left_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Heatmap (Left) - RFP Mask Individual Output", rfp_left_heatmap_colored)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(rfp_average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity (RFP Mask)')
        ax.set_title('Average GFP Intensity Heatmap \n(Left Horseshoes - RFP Mask)')
        ax.axis('off')
        plt.show()

    if all_aligned_rfp_gfp_data_right:
        rfp_right_heatmap_colored = cv2.applyColorMap(rfp_average_heatmap_right_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Heatmap (Right) - RFP Mask Individual Output", rfp_right_heatmap_colored)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(rfp_average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity (RFP Mask)')
        ax.set_title('Average GFP Intensity Heatmap \n(Right Horseshoes - RFP Mask)')
        ax.axis('off')
        plt.show()

cv2.waitKey(0) 
cv2.destroyAllWindows()

# --- Matplotlibによる統合プロット (Normal + RFP) ---

# Matplotlibで統合プロットを作成 (視覚的な確認用)
num_plots = 3 # デバッグ、左ヒートマップ、右ヒートマップの3列

num_image_sets = len(image_pairs) # processed_images_for_plotの数
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 # 少なくとも1行確保

fig = plt.figure(figsize=(num_plots * 4, 4))

plot_index = 1

rows_inner = 2
cols_inner = 2
num_debug_images = len(debug_images)

# fig = plt.figure(figsize=(num_plots * 4, 4 * 2)) # 図のサイズを調整

# ----------------------------------------------------
# 1. 外側の GridSpec を定義 (図全体を1行 num_plots列に分割)
# ----------------------------------------------------
gs_outer = gridspec.GridSpec(1, num_plots, figure=fig)
plot_index = 0

# ----------------------------------------------------
# 2. 1区画目 (左端: gs_outer[0, 0]) にデバッグ画像をプロット
# ----------------------------------------------------
if num_debug_images >= 4:
    # ----------------------------------------------------
    # 2. 外側の GridSpec を定義 (図全体を1行 num_plots列に分割)
    # これが「三分割したうち」のベースとなるグリッドです。
    gs_outer = gridspec.GridSpec(1, num_plots, figure=fig)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # 3. 1区画目 (左端: gs_outer[0, 0]) の中に、内側の GridSpec (2行2列) をネストして定義
    gs_inner = gridspec.GridSpecFromSubplotSpec(rows_inner, cols_inner, 
                                                subplot_spec=gs_outer[0, 0],
                                                wspace=0.1, hspace=0.1) # 余白の調整
    # ----------------------------------------------------

    # 4. 2x2 の各セルにデバッグ画像をプロット
    for i in range(num_debug_images):
        if(i>=4):
            break
        if i < rows_inner * cols_inner:
            # fig.add_subplot(gs_inner[i]) で、内側のグリッドのセルを指定
            ax = fig.add_subplot(gs_inner[i])

            # 画像表示処理 (元のコードを踏襲)
            ax.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f'Annotated {i+1}', fontsize=10) # タイトルを小さくする
            ax.axis('off')
else:
    # Debug画像のプロット
    for i in range(num_image_sets):
        plt.subplot(num_rows_for_display, num_plots, i * num_plots + 1)
        plt.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Annotated Image {i+1}')
        plt.axis('off')
plot_index += 1

# ----------------------------------------------------
# 3. Normal Maskのヒートマップのプロット
# ----------------------------------------------------
# 左側のヒートマップ
ax = fig.add_subplot(gs_outer[0, plot_index])
im = ax.imshow(average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
ax.set_title('Normal Mask - Left', fontsize=10)
ax.axis('off')
# plot_index += 1

# 右側のヒートマップ
ax = fig.add_subplot(gs_outer[0, plot_index])
im = ax.imshow(average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
ax.set_title('Normal Mask - Right', fontsize=10)
ax.axis('off')
# plot_index += 1

# ----------------------------------------------------
# 4. RFP Maskのヒートマップのプロット
# ----------------------------------------------------
# 左側のヒートマップ
ax = fig.add_subplot(gs_outer[0, plot_index])
im_rfp_left = ax.imshow(rfp_average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
ax.set_title('RFP Mask - Left', fontsize=10)
ax.axis('off')
plot_index += 1

# 右側のヒートマップ
ax = fig.add_subplot(gs_outer[0, plot_index])
im_rfp_right = ax.imshow(rfp_average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
ax.set_title('RFP Mask - Right', fontsize=10)
ax.axis('off')
plot_index += 1

# ----------------------------------------------------
# 5. カラーバーの追加（NormalとRFPで共通のスケールを使用）
# ----------------------------------------------------
# 最後のプロット（RFP Right）を使ってカラーバーを表示
# cbar = fig.colorbar(im_rfp_right, ax=fig.get_axes()[1:], 
#                     label='Normalized Average GFP Intensity (Common Scale)', 
#                     orientation='vertical', shrink=0.8, pad=0.01)


plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 【ヒートマップ4枚の2行2列配置】
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 4), constrained_layout=True)
cmap = 'viridis'
vmin = 0
vmax = 255

## 1. Normal Mask - Left (1行目, 1列目)
im1 = axes[0, 0].imshow(average_heatmap_left_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0, 0].set_title('Normal Mask - Ventral Side', fontsize=12)
axes[0, 0].axis('off')

## 2. Normal Mask - Right (1行目, 2列目)
im2 = axes[0, 1].imshow(average_heatmap_right_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
axes[0, 1].set_title('Normal Mask - Dorsal Side', fontsize=12)
axes[0, 1].axis('off')

## 3. RFP Mask - Left (2行目, 1列目)
im3 = axes[1, 0].imshow(rfp_average_heatmap_left_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1, 0].set_title('RFP Mask - Ventral Side', fontsize=12)
axes[1, 0].axis('off')

## 4. RFP Mask - Right (2行目, 2列目)
# カラーバーの基準として使用するため、im4を変数に格納
im4 = axes[1, 1].imshow(rfp_average_heatmap_right_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1, 1].set_title('RFP Mask - Dorsal Side', fontsize=12)
axes[1, 1].axis('off')

# 共通のカラーバーを追加 (最後のプロット im4 を基準とする)
# 左右に配置されたサブプロット全体の右側にカラーバーを配置
fig.colorbar(im4, ax=axes.ravel().tolist(), 
             label=f'Normalized Average GFP Intensity (Max = {max_gfp_intensity_normal:.2f})', 
             orientation='vertical', shrink=0.75)

# タイトル
# fig.suptitle('Comparison of GFP Intensity Heatmaps (Normal vs RFP Mask)', fontsize=14, y=1.02)

plt.show()

cv2.destroyAllWindows()