import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.gridspec as gridspec

# 除外する中心線の幅を定義 30ピクセル
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
    # 【修正1: 分割基点の変更】
    # 緑色領域ではなく、**画像全体の中心**を分割線とする
    height, width = image.shape[:2]
    image_center_x = width // 2 
    print(f"分割中心X座標: {image_center_x} (画像の中心)")
    return image_center_x

# --- メイン処理 ---
# image_pairs = [
#     {'original': './pics/control/Projections of 241106_Fz_002_second.png', 
#      'mask': './pics/control/Projections of 241106_Fz_002_second_mask.tif'},
#     {'original':'./pics/control/Projections of 241106_Fz_control003.png', 
#      'mask': './pics/control/Projections of 241106_Fz_control003_mask.tif'},
#     {'original':'./pics/control/Projections of 20241024_controlFz001.png', 
#      'mask': './pics/control/Projections of 20241024_controlFz001_mask.tif'},
#     {'original':'./pics/control/Projections of 20241024_controlFz003.png', 
#      'mask': './pics/control/Projections of 20241024_controlFz003_mask.tif'}
# ]

image_pairs = [
    {'original': './pics/vang/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue.png', 
     'mask': './pics/vang/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita.png', 
     'mask': './pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita_mask.tif'},
    {'original':'./pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue.png', 
     'mask': './pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2.png', 
     'mask': './pics/vang/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2_mask.tif'}
]

# image_pairs = [
    # {'original': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/C1-Projections of 20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063.png', 
    #  'mask': './pics/mosaic_fz_Fz/20250402_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_normal.tif'},
    # {'original':'./pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/C1-Projections of 20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063.png', 
    #  'mask': './pics/mosaic_fz_Fz/20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063/Merged_normal.tif'},
# ]

# image_pairs = [
#     {'original': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/C1-Projections of 20250423_24%APF_Fz-GFP_RFP_ue.png', 
#      'mask': './pics/mosaic_control/20250423_24%APF_Fz-GFP_RFP_ue/Merged_rfp.tif'},
#     {'original':'./pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/C1-Projections of 20250428_24%APF_Fz-GFP_sensRFP.png', 
#      'mask': './pics/mosaic_control/20250428_24%APF_Fz-GFP_sensRFP/Merged_rfp.tif'},
# ]

target_size = (100, 100) 

# =========================================================================
# 【STEP 1】 全体の基準平均輝度 (global_mean_intensity) の計算
# =========================================================================
all_raw_gfp_values = [] 

for file_info in image_pairs:
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']
    
    image_original_for_gfp = cv2.imread(original_file_name)
    image_mask_for_contours = cv2.imread(mask_file_name)
    
    if image_original_for_gfp is None or image_mask_for_contours is None:
        continue
        
    gfp_channel = image_original_for_gfp[:, :, 1]
    
    # 全ての色のマスクを結合
    combined_mask = np.zeros_like(gfp_channel, dtype=np.uint8)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        combined_mask = cv2.bitwise_or(combined_mask, extract_color_mask(image_mask_for_contours, color_name))
    
    # 結合マスク内のGFP輝度値を取得
    masked_gfp = cv2.bitwise_and(gfp_channel, gfp_channel, mask=combined_mask)
    non_zero_gfp = masked_gfp[masked_gfp > 0]
    
    all_raw_gfp_values.extend(non_zero_gfp.flatten())

# 全体の基準平均輝度 (Global Mean Intensity) を計算
if all_raw_gfp_values:
    global_mean_intensity = np.mean(all_raw_gfp_values) 
else:
    global_mean_intensity = 1.0 # データがない場合は0割を防ぐ
print(f"✅ 全体の基準平均輝度 (Global Mean): {global_mean_intensity:.2f}")


# =========================================================================
# 【STEP 2】 個々の馬蹄形データの抽出と輝度正規化
# =========================================================================
all_aligned_gfp_data_left = []
all_aligned_gfp_data_right = []
debug_images = []

# 【修正2: 元画像のリサイズをしない】
# ヒートマップのサイズは固定 (100x100) を維持
target_size = (100, 100) 

for index, file_info in enumerate(image_pairs):
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']

    # 1. オリジナル画像（輝度測定用）の読み込み
    # 【修正2: 元画像のリサイズをしない】元の輝度情報をそのまま使うため、リサイズしない
    image_original_for_gfp = cv2.imread(original_file_name)
    if image_original_for_gfp is None:
        print(f"オリジナル画像を読み込めませんでした: {original_file_name}。スキップします。")
        continue

    # 2. マスク画像（輪郭検出用）の読み込み
    image_mask_for_contours = cv2.imread(mask_file_name)
    
    if image_original_for_gfp is None or image_mask_for_contours is None:
        print(f"画像を読み込めませんでした: {original_file_name} または {mask_file_name}。スキップします。")
        continue

    # 左右分割の中心X座標を取得
    image_center_x = split_left_right(image_mask_for_contours)
    gfp_channel = image_original_for_gfp[:, :, 1] # GFPチャネル（緑）
    
    # ---------------------------------------------------------------------
    # 現在の画像の平均輝度 (current_image_mean_intensity) の計算
    # ※ STEP 1 のループと同じ処理だが、ここでは輪郭検出に使う combined_mask を再利用
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
    print(f"   - 画像の平均輝度: {current_image_mean_intensity:.2f}, スケーリングファクター: {scaling_factor:.3f}")
    # ---------------------------------------------------------------------
    # debug表示用に画像をコピー（輪郭検出用のマスク画像がベース）
    # 【修正3: アノテーション画像の個別出力のベース】
    # マスク画像ではなく、**オリジナル画像**をベースとして、その上にアノテーションの色を重ねる
    debug_img = image_original_for_gfp.copy()
    
    # 中心線を白で描画
    cv2.line(debug_img, (image_center_x, 0), (image_center_x, debug_img.shape[0]), (255, 255, 255), 1)

    all_horseshoe_masks = []
    # extract_color_maskを使って、馬蹄形輪郭を抽出 (全ての定義済み色を結合)
    for color_name in ANNOTATION_COLORS_HSV.keys():
        horseshoe_mask = extract_color_mask(image_mask_for_contours, color_name)
        all_horseshoe_masks.append(horseshoe_mask)

    # 各輪郭を左右に分類し、GFP輝度データを抽出
    valid_horseshoe_count_left = 0
    valid_horseshoe_count_right = 0
    
    # 【修正2: 元画像のリサイズをしない】
    # GFPチャネルを抽出（輝度測定には**オリジナル画像**を使用、緑チャネルをGFP輝度とする）
    gfp_channel = image_original_for_gfp[:, :, 1] # BGR画像の2番目(インデックス1)がG(緑)チャネル

    # all_horseshoe_masks の各マスクをループ
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
            
            # 【修正1: 分割基点の変更】
            threashold = 10 # 中心線からの無視する距離を小さくする（データが中心に集中していない場合）
            
            is_overlapping = (x < image_center_x < x + w)
            if is_overlapping:
                # 中心線に被る輪郭は無視
                color = (0, 255, 255) # 黄色 (BGR) for ignored contours
                # cv2.drawContours(debug_img, [contour], -1, color, -1) 
                continue 
            elif contour_center_x + threashold < image_center_x:
                # 左側の輪郭
                current_gfp_data_list = all_aligned_gfp_data_left
                # 【修正3: 赤/青アノテーション】左側は赤
                color = (0, 0, 255) # 赤 (BGR) for debug
                valid_horseshoe_count_left += 1
            elif contour_center_x - threashold > image_center_x:
                # 右側の輪郭
                current_gfp_data_list = all_aligned_gfp_data_right
                # 【修正3: 赤/青アノテーション】右側は青
                color = (255, 0, 0) # 青 (BGR) for debug
                valid_horseshoe_count_right += 1
            else:
                color = (0, 255, 255) # 黄色 (BGR) for ignored contours
                # cv2.drawContours(debug_img, [contour], -1, color, -1) 
                continue

            # 膨張処理は元のコードのロジックを維持 (占有率に基づく処理)
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            rel_contour = contour - np.array([x, y])
            cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)
            
            kernel = np.ones((3, 3), np.uint8)
            iterations = 0
            contour_mask_local = temp_mask.copy()
            
            # 膨張ロジック（元のコードのまま）
            # ... (中略: 膨張ロジックは変更なし)
            while True:
                total_pixels = w * h
                white_pixels = cv2.countNonZero(contour_mask_local)
                occupancy_rate = white_pixels / total_pixels if total_pixels > 0 else 0
                
                if occupancy_rate > 0.60 or iterations >= 4:
                    break

                contour_mask_local = cv2.dilate(contour_mask_local, kernel, iterations=1)
                iterations += 1

            if iterations > 2: # 膨張が3回以上ならスキップ (元のコードのロジックを維持)
                 continue
            
            # 最終的な輪郭を取得し、debug_imgに描画
            final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                final_absolute_contour = final_contours[0] + np.array([x, y])
                # 【修正3: アノテーション画像の個別出力】検出した輪郭をdebug_img（オリジナル画像ベース）に赤/青で描画
                cv2.drawContours(debug_img, [final_absolute_contour], -1, color, -1)
    
            # 元のGFPチャネル画像（オリジナル画像から抽出）から、このバウンディングボックスに対応する部分を切り出す
            gfp_roi = gfp_channel[y:y+h, x:x+w]

            # 切り出したGFP輝度データに、作成したローカルマスクを適用
            masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)

            # リサイズ (target_sizeに統一) - ヒートマップの平均化のためにリサイズは必要
            resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
            
            # ---------------------------------------------------------------------
            # 【重要】輝度正規化の適用
            # D' = D * (I_global_mean / I_current_image_mean)
            # ---------------------------------------------------------------------
            resized_gfp_float = resized_gfp.astype(np.float64) 
            resized_gfp_normalized_float = resized_gfp_float * scaling_factor
            
            # 結果を uint8 に戻し、クリッピング
            # このデータを使って後の平均化を行う
            resized_gfp_final = np.clip(resized_gfp_normalized_float, 0, 255).astype(np.uint8)
            
            current_gfp_data_list.append(resized_gfp_final)
            
    cv2.imshow(f"Annotated Image: {os.path.basename(original_file_name)}", debug_img)
    cv2.waitKey(1) # 一瞬表示させるため

    debug_images.append(debug_img)
    print(f"'{original_file_name}' から左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")

    if(index == 3 & DEBUG_FLAG == 1):
        # 保存用のヒートマップの表示
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True) # figsizeで図のサイズを指定できます
        im = ax.imshow(debug_img, cmap='viridis', vmin=0, vmax=255)
        # 軸の非表示
        ax.axis('off')
        # 図の表示
        plt.show()


### 4. すべての輝度データの平均化とヒートマップの個別出力

average_heatmap_left_normalized = np.zeros(target_size, dtype=np.uint8)
if all_aligned_gfp_data_left:
    average_heatmap_left = np.mean(np.array(all_aligned_gfp_data_left), axis=0)
    average_heatmap_left_normalized = cv2.normalize(average_heatmap_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("左側の有効な馬蹄形が全く検出されませんでした。左側のヒートマップは黒になります。")

average_heatmap_right_normalized = np.zeros(target_size, dtype=np.uint8)
if all_aligned_gfp_data_right:
    average_heatmap_right = np.mean(np.array(all_aligned_gfp_data_right), axis=0)
    average_heatmap_right_normalized = cv2.normalize(average_heatmap_right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("右側の有効な馬蹄形が全く検出されませんでした。右側のヒートマップは黒になります。")

if(DEBUG_FLAG == 1):
    # 【修正4: Heatmapの個別出力】
    if all_aligned_gfp_data_left:
        # BGRまたはRGB画像に変換しないとimshowはカラーマップを適用できない
        # 実際にはmatplotlibのcmapを使いたいが、cv2.imshowを使う場合は、擬似カラーを適用する
        left_heatmap_colored = cv2.applyColorMap(average_heatmap_left_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Heatmap (Left) - Individual Output", left_heatmap_colored)

        # 保存用のヒートマップの表示
        fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # figsizeで図のサイズを指定できます
        im = ax.imshow(average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
        # カラーバーの追加
        plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity')
        # タイトルの設定
        ax.set_title('Average GFP Intensity Heatmap \n(Left Horseshoes)')
        # 軸の非表示
        ax.axis('off')
        # 図の表示
        plt.show()

    if all_aligned_gfp_data_right:
        right_heatmap_colored = cv2.applyColorMap(average_heatmap_right_normalized, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Heatmap (Right) - Individual Output", right_heatmap_colored)

        # 保存用のヒートマップの表示
        fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # figsizeで図のサイズを指定できます
        im = ax.imshow(average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
        # カラーバーの追加
        plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity')
        # タイトルの設定
        ax.set_title('Average GFP Intensity Heatmap \n(Left Horseshoes)')
        # 軸の非表示
        ax.axis('off')
        # 図の表示
        plt.show()

cv2.waitKey(0) 
cv2.destroyAllWindows()


# --- Matplotlibによる統合プロット (元のコードの維持) ---

# Matplotlibで統合プロットを作成 (視覚的な確認用)
num_plots = 3 # デバッグ、左ヒートマップ、右ヒートマップの3列

num_image_sets = len(image_pairs) # processed_images_for_plotの数
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 # 少なくとも1行確保
num_plots = 3
num_image_sets = len(image_pairs) 
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 

fig = plt.figure(figsize=(num_plots * 4, 4))

plot_index = 1

rows_inner = 2
cols_inner = 2
num_debug_images = len(debug_images)

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

# 左側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Ventral Horseshoes)')
plt.axis('off')
plot_index += 1

# 右側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Dorsal Horseshoes)')
plt.axis('off')
plot_index += 1

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()