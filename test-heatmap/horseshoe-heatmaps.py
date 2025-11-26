import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

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

def split_left_right(image):
    # 【修正1: 分割基点の変更】
    # 緑色領域ではなく、**画像全体の中心**を分割線とする
    height, width = image.shape[:2]
    image_center_x = width // 2 
    print(f"分割中心X座標: {image_center_x} (画像の中心)")
    return image_center_x

# --- メイン処理 ---
image_pairs = [
    {'original': './pics/Projections of 241106_Fz_002_second.png', 
     'mask': './pics/Projections of 241106_Fz_002_second_mask.tif'},
    {'original':'./pics/Projections of 241106_Fz_control003.png', 
     'mask': './pics/Projections of 241106_Fz_control003_mask.tif'},
    {'original':'./pics/Projections of 20241024_controlFz001.png', 
     'mask': './pics/Projections of 20241024_controlFz001_mask.tif'},
    {'original':'./pics/Projections of 20241024_controlFz003.png', 
     'mask': './pics/Projections of 20241024_controlFz003_mask.tif'}
]

all_aligned_gfp_data_left = []
all_aligned_gfp_data_right = []
debug_images = []

# 【修正2: 元画像のリサイズをしない】
# ヒートマップのサイズは固定 (100x100) を維持
target_size = (100, 100) 

for file_info in image_pairs:
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
    if image_mask_for_contours is None:
        print(f"マスク画像を読み込めませんでした: {mask_file_name}。スキップします。")
        continue

    # 左右分割の中心X座標を取得
    image_center_x = split_left_right(image_mask_for_contours)

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
            resized_gfp = np.clip(resized_gfp, 0, 255).astype(np.uint8) 

            current_gfp_data_list.append(resized_gfp)

    # 【修正3: アノテーション画像の個別出力】
    # 処理された画像ごとに、アノテーションを赤/青で示した画像を個別に出力
    cv2.imshow(f"Annotated Image: {os.path.basename(original_file_name)}", debug_img)
    cv2.waitKey(1) # 一瞬表示させるため

    debug_images.append(debug_img)
    print(f"'{original_file_name}' から左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")


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

# 【修正4: Heatmapの個別出力】
if all_aligned_gfp_data_left:
    # BGRまたはRGB画像に変換しないとimshowはカラーマップを適用できない
    # 実際にはmatplotlibのcmapを使いたいが、cv2.imshowを使う場合は、擬似カラーを適用する
    left_heatmap_colored = cv2.applyColorMap(average_heatmap_left_normalized, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Heatmap (Left) - Individual Output", left_heatmap_colored)

if all_aligned_gfp_data_right:
    right_heatmap_colored = cv2.applyColorMap(average_heatmap_right_normalized, cv2.COLORMAP_VIRIDIS)
    cv2.imshow("Heatmap (Right) - Individual Output", right_heatmap_colored)

cv2.waitKey(0) 
cv2.destroyAllWindows()


# --- Matplotlibによる統合プロット (元のコードの維持) ---

# Matplotlibで統合プロットを作成 (視覚的な確認用)
num_plots = 3 # デバッグ、左ヒートマップ、右ヒートマップの3列

num_image_sets = len(image_pairs) # processed_images_for_plotの数
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 # 少なくとも1行確保
num_plots = 3

plt.figure(figsize=(num_plots * 4, 7)) # プロット数に応じて全体のサイズを調整

plot_index = 1

# # Debug画像のプロット
for i in range(num_image_sets):
    # オリジナル画像のプロット (i行目の1列目)
    plt.subplot(num_rows_for_display, num_plots, i * num_plots + 1)
    plt.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image {i+1} \n(Cropped)')
    plt.axis('off')
plot_index += 1

# 左側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(average_heatmap_left_normalized, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Left Horseshoes)')
plt.axis('off')
plot_index += 1

# 右側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(average_heatmap_right_normalized, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Right Horseshoes)')
plt.axis('off')
plot_index += 1

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()