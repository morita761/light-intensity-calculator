import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 除外する中心線の幅を定義 30ピクセル
EXCLUSION_WIDTH = 30

# 極座標ヒストグラムのビンの幅
BIN_NUM = 37


# アノテーションに使用する色のHSV範囲を定義 (変更なし)
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

def extract_color_mask(mask_image_bgr, color_name):
    """指定された色のHSV範囲に基づいてマスクを抽出する (変更なし)"""
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
    """画像全体の中心X座標を左右分割線として返す (変更なし)"""
    height, width = image.shape[:2]
    image_center_x = width // 2 
    print(f"分割中心X座標: {image_center_x} (画像の中心)")
    return image_center_x

# ----------------------------------------------------
# 【維持】凹の方向を計算する関数 (下=0度、右=90度、左=-90度)
# ----------------------------------------------------
def calculate_horseshoe_orientation(contour):
    """
    馬蹄形輪郭から凹（開口部）の方向を推定し、画像下=0度、右=90度、左=-90度の角度を返す。
    
    ロジック: 重心Cからバウンディングボックスの中心Bへ向かうベクトル (C -> B) を凹の方向とする。
    """
    if len(contour) < 3:
        return None

    # 1. 輪郭のモーメントを計算し、重心 (Centroid: Cx, Cy) を求める
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    
    # 重心 C
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 2. バウンディングボックスの中心 (Box Center: Bx, By) を求める
    x, y, w, h = cv2.boundingRect(contour)
    bx = x + w // 2
    by = y + h // 2

    # 3. 凹の方向ベクトルを計算: V = B - C
    # V_x = Bx - Cx
    # V_y = By - Cy
    
    # 4. 角度の計算
    # atan2(y, x) は標準的なデカルト座標系（右が0度、上が90度）で角度を返す
    # 注意: 画像座標系はy軸が下向き正であるため、y座標を反転させて標準的なデカルト座標系に合わせる
    # V_y を反転: -(By - Cy) = Cy - By
    
    angle_rad_standard = np.arctan2(cy - by, bx - cx)
    angle_deg_standard = np.degrees(angle_rad_standard)
    
    # 5. 座標系の変換: 要件 (下=0度、右=90度)
    # 新しい角度 = 標準角度 + 90
    orientation_angle = angle_deg_standard + 90
    
    # 角度を [-180, 180] の範囲に正規化
    if orientation_angle > 180:
        orientation_angle -= 360
    elif orientation_angle <= -180:
        orientation_angle += 360
    
    # デバッグ表示用に、ここではバウンディングボックスの中心 (bx, by) を開口部側の代表点として返す
    return orientation_angle, cx, cy, bx, by
# ----------------------------------------------------

# --- メイン処理 ---
image_pairs = [
    {'original': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl_5degree_24hours_senslexAlexOP_loco_cas9_contorl_rot.tif', 
     'mask': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl_5degree_24hours_senslexAlexOP_loco_cas9_contorl_rot_detection.tif'},
]

# image_pairs = [
#     {'original': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001.png', 
#      'mask': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001_detection.tif'},
# ]

debug_images = []
left_horseshoe_angles = []
right_horseshoe_angles = []

# 【新規追加】個々の馬蹄形のデバッグ画像を格納するリスト
individual_debug_plots = [] 

for index, file_info in enumerate(image_pairs):
    original_file_name = file_info['original']
    mask_file_name = file_info['mask']

    image_original_for_gfp = cv2.imread(original_file_name)
    if image_original_for_gfp is None:
        print(f"オリジナル画像を読み込めませんでした: {original_file_name}。スキップします。")
        continue

    image_mask_for_contours = cv2.imread(mask_file_name)
    if image_mask_for_contours is None:
        print(f"マスク画像を読み込めませんでした: {mask_file_name}。スキップします。")
        continue

    image_center_x = split_left_right(image_mask_for_contours)

    debug_img = image_original_for_gfp.copy()
    cv2.line(debug_img, (image_center_x, 0), (image_center_x, debug_img.shape[0]), (255, 255, 255), 1)

    all_horseshoe_masks = []
    for color_name in ANNOTATION_COLORS_HSV.keys():
        horseshoe_mask = extract_color_mask(image_mask_for_contours, color_name)
        all_horseshoe_masks.append(horseshoe_mask)

    valid_horseshoe_count_left = 0
    valid_horseshoe_count_right = 0
    
    gfp_channel = image_original_for_gfp[:, :, 1] 

    for single_horseshoe_mask in all_horseshoe_masks:
        contours, _ = cv2.findContours(single_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 20:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_x = x + w // 2
            
            is_overlapping = (x < image_center_x < x + w)
            if is_overlapping:
                continue 
            
            current_angle_list = None
            side_label = ""
            if contour_center_x + EXCLUSION_WIDTH < image_center_x:
                # 左側の輪郭
                color = (0, 0, 255) # 赤 (BGR) for debug
                valid_horseshoe_count_left += 1
                current_angle_list = left_horseshoe_angles
                side_label = "Left"
            elif contour_center_x - EXCLUSION_WIDTH > image_center_x:
                # 右側の輪郭
                color = (255, 0, 0) # 青 (BGR) for debug
                valid_horseshoe_count_right += 1
                current_angle_list = right_horseshoe_angles
                side_label = "Right"
            else:
                continue

            orientation_result = calculate_horseshoe_orientation(contour)
            if orientation_result:
                angle, cx, cy, fx, fy = orientation_result
                current_angle_list.append(angle)
                
                # デバッグ表示: 重心(白)、開口部(緑)、方向ベクトル(シアン)
                cv2.circle(debug_img, (cx, cy), 3, (255, 255, 255), -1) 
                cv2.circle(debug_img, (fx, fy), 3, (0, 255, 0), -1)   
                cv2.line(debug_img, (cx, cy), (fx, fy), (255, 255, 0), 1) 

                # ----------------------------------------------------
                # ✨【新規追加】個々の馬蹄形のデバッグ表示
                # ----------------------------------------------------
                
                # 1. バウンディングボックス領域を切り抜き (ROI)
                roi = image_original_for_gfp[y:y+h, x:x+w].copy()

                # 2. ROI内に重心(cx_local, cy_local)と最遠点(fx_local, fy_local)を計算
                cx_local = cx - x
                cy_local = cy - y
                fx_local = fx - x
                fy_local = fy - y

                # 3. 拡大表示のためにリサイズ
                display_size = (150, 150)
                roi_resized = cv2.resize(roi, display_size, interpolation=cv2.INTER_LINEAR)
                
                # 4. 拡大率を考慮して、ベクトルを再計算
                scale_x = display_size[0] / w
                scale_y = display_size[1] / h

                cx_disp = int(cx_local * scale_x)
                cy_disp = int(cy_local * scale_y)
                fx_disp = int(fx_local * scale_x)
                fy_disp = int(fy_local * scale_y)

                # 5. 切り抜き画像に矢印を描画
                # 矢印は重心から開口部の方向 (凹の方向) へ
                cv2.arrowedLine(roi_resized, (cx_disp, cy_disp), (fx_disp, fy_disp), (0, 255, 255), 2, tipLength=0.3)
                
                # 6. 角度テキストの準備
                angle_text = f"Angle: {angle:.1f}°"
                
                individual_debug_plots.append({
                    'image': cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB),
                    'angle': angle_text,
                    'side': side_label
                })
                # ----------------------------------------------------

            # 膨張処理は元のコードのロジックを維持 (中略)
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
            
            final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                final_absolute_contour = final_contours[0] + np.array([x, y])
                cv2.drawContours(debug_img, [final_absolute_contour], -1, color, -1)
            
    cv2.imshow(f"Annotated Image: {os.path.basename(original_file_name)}", debug_img)
    cv2.waitKey(1) 

    debug_images.append(debug_img)
    print(f"'{original_file_name}' から左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")

cv2.waitKey(0) 
cv2.destroyAllWindows()

# ----------------------------------------------------
## 🐎 個別馬蹄形デバッグ表示
# ----------------------------------------------------
# individual_debug_plots = [] # debugを省略：使用するときはこれをコメントアウト
if individual_debug_plots:
    # 10個ごとに新しいウィンドウを作成して表示
    plots_per_figure = 10
    num_figures = (len(individual_debug_plots) + plots_per_figure - 1) // plots_per_figure
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * plots_per_figure
        end_idx = min((fig_idx + 1) * plots_per_figure, len(individual_debug_plots))
        current_plots = individual_debug_plots[start_idx:end_idx]

        num_cols = 5
        num_rows = (len(current_plots) + num_cols - 1) // num_cols
        
        plt.figure(figsize=(num_cols * 3, num_rows * 3))
        plt.suptitle(f"Individual Horseshoe Debug Plots (Set {fig_idx + 1})", fontsize=16)

        for i, plot_data in enumerate(current_plots):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(plot_data['image'])
            plt.title(f"{plot_data['side']} | {plot_data['angle']}", fontsize=10)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルとの干渉を避ける
        plt.show()
else:
    print("\n個々の馬蹄形のデバッグデータは収集されませんでした。")


# ----------------------------------------------------
## 📊 統合プロット (メイン画像と極座標ヒストグラム)
# ----------------------------------------------------

num_plots = 3
num_image_sets = len(image_pairs) 
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 

plt.figure(figsize=(num_plots * 4, 7))

# Debug画像のプロット
for i in range(num_image_sets):
    plt.subplot(num_rows_for_display, num_plots, i * num_plots + 1)
    plt.imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f'Annotated Image {i+1}')
    plt.axis('off')

# 左側の極座標ヒストグラム
ax_left = plt.subplot(1, num_plots, 2, projection='polar')
if left_horseshoe_angles:
    left_angles_rad = np.radians(left_horseshoe_angles)
    angle_range = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False) 
    counts, bin_edges = np.histogram(left_angles_rad, bins=angle_range)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)
    
    bars = ax_left.bar(centers, counts, width=width, color='red', alpha=0.6, bottom=0)
    
    ax_left.set_theta_zero_location("S") 
    ax_left.set_rlim(0, max(counts) * 1.1 if counts.size > 0 else 1)
    
    ax_left.set_xticks(np.radians([0, 90, 180, 270]))
    ax_left.set_xticklabels(['0°', '90°', '180°', ' -90°'])
    ax_left.set_title('Orientation Polar Histogram (Left)', va='bottom')
else:
    ax_left.text(0, 0, 'No data', ha='center', va='center', transform=ax_left.transAxes)

# 右側の極座標ヒストグラム
ax_right = plt.subplot(1, num_plots, 3, projection='polar')
if right_horseshoe_angles:
    right_angles_rad = np.radians(right_horseshoe_angles)
    angle_range = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False) 
    counts, bin_edges = np.histogram(right_angles_rad, bins=angle_range)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)

    bars = ax_right.bar(centers, counts, width=width, color='blue', alpha=0.6, bottom=0)
    
    ax_right.set_theta_zero_location("S")
    ax_right.set_rlim(0, max(counts) * 1.1 if counts.size > 0 else 1)
    
    ax_right.set_xticks(np.radians([0, 90, 180, 270]))
    ax_right.set_xticklabels(['0°', '90°', '180°', ' -90°'])
    ax_right.set_title('Orientation Polar Histogram (Right)', va='bottom')
else:
    ax_right.text(0, 0, 'No data', ha='center', va='center', transform=ax_right.transAxes)

plt.tight_layout()
plt.show()