import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.gridspec as gridspec

# 除外する中心線の幅を定義 30ピクセル
EXCLUSION_WIDTH = 50

# 極座標ヒストグラムのビンの幅 (-0.5° ~ 0.5°にするため)
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
# 【改善版】輪郭1Dパラメータ化を使った凹の方向を計算する関数 (下=0度、右=90度、左=-90度)
# ----------------------------------------------------
def find_opening_endpoints_by_contour(contour):
    """
    輪郭の1Dパラメータ化により、馬蹄形の開口部端点を検出する。

    アルゴリズム:
    1. 輪郭点を弧長順に取得（閉曲線として扱う）
    2. 凸包の重心を計算
    3. 各輪郭点から重心までの距離を計算（距離プロファイル）
    4. 距離が大きい連続領域 = 開口部
    5. その領域の両端を開口部端点として返す
    """
    # 輪郭を (N, 2) 形式に変換
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)

    if n < 10:
        return None

    # 凸包の重心を計算
    hull_points = cv2.convexHull(contour, returnPoints=True)
    M_hull = cv2.moments(hull_points)
    if M_hull["m00"] == 0:
        return None
    hull_cx = M_hull["m10"] / M_hull["m00"]
    hull_cy = M_hull["m01"] / M_hull["m00"]

    # 各輪郭点から重心までの距離を計算
    distances = np.sqrt((pts[:, 0] - hull_cx)**2 + (pts[:, 1] - hull_cy)**2)

    # 距離の閾値を設定（中央値より上の点を「遠い」と判定）
    threshold = np.percentile(distances, 60)

    # 閾値以上の点をマーク
    is_far = distances >= threshold

    # 連続する「遠い」領域を検出（閉曲線なので循環を考慮）
    # まず、is_far配列を2倍にして循環を処理
    is_far_extended = np.concatenate([is_far, is_far])

    # 連続領域を検出
    regions = []
    in_region = False
    start_idx = 0

    for i in range(len(is_far_extended)):
        if is_far_extended[i] and not in_region:
            in_region = True
            start_idx = i
        elif not is_far_extended[i] and in_region:
            in_region = False
            end_idx = i - 1
            # n を超える領域は最初の n 点に収まる部分のみ考慮
            if start_idx < n:
                regions.append((start_idx, end_idx, end_idx - start_idx + 1))

    # 最後の領域が閉じていない場合
    if in_region and start_idx < n:
        regions.append((start_idx, len(is_far_extended) - 1, len(is_far_extended) - start_idx))

    if not regions:
        return None

    # 最も長い連続領域を選択
    regions.sort(key=lambda x: x[2], reverse=True)
    best_region = regions[0]
    start, end, length = best_region

    # 領域の両端のインデックスを取得（循環を考慮）
    start_idx = start % n
    end_idx = end % n

    # 端点の座標を取得
    ep1 = pts[start_idx]
    ep2 = pts[end_idx]

    return ep1, ep2, (hull_cx, hull_cy)


def calculate_horseshoe_orientation(contour):
    """
    馬蹄形輪郭から凹（開口部）の方向を推定し、画像下=0度、右=90度、左=-90度の角度を返す。

    ロジック: 輪郭の1Dパラメータ化により開口部の端点を検出し、
    凸包重心→端点中点の方向ベクトルで向きを計算する。
    """
    if len(contour) < 5:
        return None

    # 1. 輪郭ベースで開口部端点を検出
    result = find_opening_endpoints_by_contour(contour)

    if result is None:
        return fallback_orientation(contour)

    ep1, ep2, (hull_cx, hull_cy) = result

    # 2. 端点の中点を計算（開口部の中心）
    midpoint_x = (ep1[0] + ep2[0]) / 2
    midpoint_y = (ep1[1] + ep2[1]) / 2

    # 3. 方向ベクトル: 凸包重心 → 端点中点（開口部方向）
    dx = midpoint_x - hull_cx
    dy = midpoint_y - hull_cy

    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return fallback_orientation(contour)

    # 正規化
    dx_norm = dx / norm
    dy_norm = dy / norm

    # 4. 始点を端点中点に設定し、そこから外側へ矢印を延ばす
    x, y, w, h = cv2.boundingRect(contour)
    cx = int(midpoint_x)
    cy = int(midpoint_y)

    arrow_length = max(w, h) * 0.6
    fx = int(cx + dx_norm * arrow_length)
    fy = int(cy + dy_norm * arrow_length)

    # 5. 角度の計算
    angle_rad_standard = np.arctan2(-dy, dx)
    angle_deg_standard = np.degrees(angle_rad_standard)

    # 6. 座標系の変換: 要件 (下=0度、右=90度)
    orientation_angle = angle_deg_standard + 90

    # 角度を [-180, 180] の範囲に正規化
    if orientation_angle > 180:
        orientation_angle -= 360
    elif orientation_angle <= -180:
        orientation_angle += 360

    return orientation_angle, cx, cy, fx, fy


def fallback_orientation(contour):
    """輪郭検出に失敗した場合のフォールバック（重心ベース）"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(contour)
    bx = x + w // 2
    by = y + h // 2

    angle_rad_standard = np.arctan2(cy - by, bx - cx)
    angle_deg_standard = np.degrees(angle_rad_standard)

    orientation_angle = angle_deg_standard + 90

    if orientation_angle > 180:
        orientation_angle -= 360
    elif orientation_angle <= -180:
        orientation_angle += 360

    arrow_length = max(w, h) * 0.8
    dx = bx - cx
    dy = by - cy
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return None
    fx = int(cx + (dx / norm) * arrow_length)
    fy = int(cy + (dy / norm) * arrow_length)

    return orientation_angle, cx, cy, fx, fy
# ----------------------------------------------------

# --- メイン処理 ---
# image_pairs = [
#     {'original': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image.png', 
#      'mask': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image_mask.tif'},
#     {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h.png', 
#      'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h_mask.tif'},
#     {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h.png', 
#      'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h_mask.tif'},
# ]

image_pairs = [
    {'original': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001.png', 
     'mask': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.png', 
     'mask': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.png', 
     'mask': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.png', 
     'mask': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001_mask.tif'},
]

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
                color = (255, 0, 255) # マゼンタ (BGR) for debug
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
                # 矢印を長くするために方向ベクトルを延長
                arrow_scale = 2.5  # 矢印の長さ倍率
                dx = fx - cx
                dy = fy - cy
                fx_ext = int(cx + dx * arrow_scale)
                fy_ext = int(cy + dy * arrow_scale)
                cv2.arrowedLine(debug_img, (cx, cy), (fx_ext, fy_ext), (255, 255, 0), 2, tipLength=0.2) 

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
                # 矢印を長くするために方向ベクトルを延長
                dx_disp = fx_disp - cx_disp
                dy_disp = fy_disp - cy_disp
                fx_disp_ext = int(cx_disp + dx_disp * arrow_scale)
                fy_disp_ext = int(cy_disp + dy_disp * arrow_scale)
                cv2.arrowedLine(roi_resized, (cx_disp, cy_disp), (fx_disp_ext, fy_disp_ext), (0, 255, 255), 2, tipLength=0.2)
                
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
individual_debug_plots = [] # debugを省略：使用するときはこれをコメントアウト
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

fig = plt.figure(figsize=(num_plots * 4, 4))

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

# 左側の極座標ヒストグラム
ax_left = plt.subplot(1, num_plots, 2, projection='polar')
if left_horseshoe_angles:
    left_angles_rad = np.radians(left_horseshoe_angles)
    angle_range = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False) 
    counts, bin_edges = np.histogram(left_angles_rad, bins=angle_range)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)
    
    bars = ax_left.bar(centers, counts, width=width, color='magenta', alpha=0.6, bottom=0)
    
    ax_left.set_theta_zero_location("S") 
    ax_left.set_rlim(0, max(counts) * 1.1 if counts.size > 0 else 1)
    
    ax_left.set_xticks(np.radians([0, 90, 180, 270]))
    ax_left.set_xticklabels(['0°', '90°', '180°', ' -90°'])
    ax_left .set_rlabel_position(135)
    ax_left.set_title('Orientation Polar Histogram (Ventral)', va='bottom')
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
    ax_right.set_rlabel_position(135)
    ax_right.set_title('Orientation Polar Histogram (Dorsal)', va='bottom')
else:
    ax_right.text(0, 0, 'No data', ha='center', va='center', transform=ax_right.transAxes)

plt.tight_layout()
plt.show()