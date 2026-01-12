import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.gridspec as gridspec
from skimage.morphology import skeletonize

# 除外する中心線の幅を定義 30ピクセル
EXCLUSION_WIDTH = 80

# 極座標ヒストグラムのビンの幅 (-0.5° ~ 0.5°にするため)
BIN_WIDTH_DEG = 10.0        # -5°〜+5°
BIN_NUM = int(360 / BIN_WIDTH_DEG)+1  # 180

# デバッグフラグ: True にすると骨格化・端点の詳細を表示
DEBUG_SKELETON = False


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
# 【改善版】骨格化を使った凹の方向を計算する関数 (下=0度、右=90度、左=-90度)
# ----------------------------------------------------
def find_skeleton_endpoints_and_branches(skeleton):
    """骨格画像から端点と分岐点を検出する"""
    endpoints = []
    branch_points = []
    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                # 8近傍の接続数をカウント
                neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - 1  # 自分自身を除く
                if neighbors == 1:  # 端点は接続が1つだけ
                    endpoints.append((j, i))  # (x, y)形式
                elif neighbors >= 3:  # 分岐点は接続が3つ以上
                    branch_points.append((j, i))  # (x, y)形式

    return endpoints, branch_points


def bfs_distance(skeleton, start, end):
    """
    骨格上でstartからendまでのBFS最短経路長を計算する。
    到達不可能な場合は-1を返す。
    """
    from collections import deque

    rows, cols = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
    visited[start[1], start[0]] = True

    # 8近傍の方向
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        x, y, dist = queue.popleft()

        if (x, y) == end:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if skeleton[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny, dist + 1))

    return -1  # 到達不可能


def find_longest_path_endpoints(skeleton, endpoints):
    """
    骨格グラフ上で最長パスを持つ端点ペアを見つける。
    端点が3個以上ある場合に使用する。
    """
    if len(endpoints) < 2:
        return None

    max_dist = -1
    best_pair = (endpoints[0], endpoints[1])

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dist = bfs_distance(skeleton, endpoints[i], endpoints[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (endpoints[i], endpoints[j])

    return best_pair, max_dist

def calculate_horseshoe_orientation(contour, debug=False):
    """
    馬蹄形輪郭から凹（開口部）の方向を推定し、画像下=0度、右=90度、左=-90度の角度を返す。

    ロジック: U字形を骨格化（スケルトン化）して1本の曲線にし、
    その端点から開口部方向を推定する。

    debug=Trueの場合、追加のデバッグ情報を返す。
    """
    debug_info = None

    if len(contour) < 5:
        return None if not debug else (None, None)

    # 1. バウンディングボックスを取得
    x, y, w, h = cv2.boundingRect(contour)

    # パディングを追加（骨格化の精度向上のため）
    padding = 5
    mask_w = w + 2 * padding
    mask_h = h + 2 * padding

    # 2. 輪郭を塗りつぶしたマスクを作成
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    shifted_contour = contour - np.array([x - padding, y - padding])
    cv2.drawContours(mask, [shifted_contour], -1, 255, cv2.FILLED)

    # 3. 骨格化（スケルトン化）
    skeleton = skeletonize(mask > 0)

    # 4. 骨格の端点と分岐点を検出
    endpoints, branch_points = find_skeleton_endpoints_and_branches(skeleton)

    # デバッグ情報を作成
    if debug:
        # 骨格画像をカラーに変換（可視化用）
        skeleton_vis = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        skeleton_vis[skeleton] = (255, 255, 255)  # 骨格を白で表示

        # 端点を緑で表示
        for ep in endpoints:
            cv2.circle(skeleton_vis, ep, 3, (0, 255, 0), -1)

        # 分岐点を赤で表示
        for bp in branch_points:
            cv2.circle(skeleton_vis, bp, 3, (0, 0, 255), -1)

        # マスク画像も保存
        debug_info = {
            'skeleton_vis': skeleton_vis,
            'mask': mask,
            'num_endpoints': len(endpoints),
            'num_branch_points': len(branch_points),
            'endpoints': endpoints,
            'branch_points': branch_points
        }

    if len(endpoints) < 2:
        # 端点が2つ未満の場合は元のアルゴリズムにフォールバック
        result = fallback_orientation(contour)
        if debug:
            if debug_info:
                debug_info['fallback'] = True
            return result, debug_info
        return result

    # 5. 端点の中点を計算（U字の開口部の中心）
    # 端点が2つで分岐点がない場合はそのまま使用
    # 端点が3つ以上または分岐点がある場合は最長パスで補正
    use_longest_path = len(endpoints) != 2 or len(branch_points) > 0

    if not use_longest_path:
        # 正常ケース: 端点が2つで分岐なし
        ep1, ep2 = endpoints[0], endpoints[1]
        selection_method = "direct"
    else:
        # 補正ケース: 最長パス（geodesic distance）を使用
        result = find_longest_path_endpoints(skeleton, endpoints)
        if result:
            (ep1, ep2), path_dist = result
            selection_method = f"longest_path(dist={path_dist})"
        else:
            # フォールバック: ユークリッド距離で最も離れた2点
            max_dist = 0
            ep1, ep2 = endpoints[0], endpoints[1]
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 +
                                   (endpoints[i][1] - endpoints[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        ep1, ep2 = endpoints[i], endpoints[j]
            selection_method = "euclidean_fallback"

    # 選択された端点をデバッグ情報に追加
    if debug and debug_info:
        debug_info['selected_endpoints'] = (ep1, ep2)
        debug_info['selection_method'] = selection_method
        debug_info['fallback'] = False

    # 端点の中点（ローカル座標）
    midpoint_local_x = (ep1[0] + ep2[0]) / 2
    midpoint_local_y = (ep1[1] + ep2[1]) / 2

    # グローバル座標に変換
    midpoint_x = midpoint_local_x + x - padding
    midpoint_y = midpoint_local_y + y - padding

    # 6. 凸包の重心を計算
    hull_points = cv2.convexHull(contour, returnPoints=True)
    M_hull = cv2.moments(hull_points)
    if M_hull["m00"] == 0:
        return None if not debug else (None, debug_info)
    hull_cx = int(M_hull["m10"] / M_hull["m00"])
    hull_cy = int(M_hull["m01"] / M_hull["m00"])

    # 7. 方向ベクトル: 凸包重心 → 端点中点（開口部方向）
    dx = midpoint_x - hull_cx
    dy = midpoint_y - hull_cy

    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return None if not debug else (None, debug_info)

    # 正規化
    dx_norm = dx / norm
    dy_norm = dy / norm

    # 8. 始点を端点中点に設定し、そこから外側へ矢印を延ばす
    cx = int(midpoint_x)
    cy = int(midpoint_y)

    arrow_length = max(w, h) * 0.6
    fx = int(cx + dx_norm * arrow_length)
    fy = int(cy + dy_norm * arrow_length)

    # 9. 角度の計算
    angle_rad_standard = np.arctan2(-dy, dx)
    angle_deg_standard = np.degrees(angle_rad_standard)

    # 10. 座標系の変換: 要件 (下=0度、右=90度)
    orientation_angle = angle_deg_standard + 90

    # 角度を [-180, 180] の範囲に正規化
    if orientation_angle > 180:
        orientation_angle -= 360
    elif orientation_angle <= -180:
        orientation_angle += 360

    result = (orientation_angle, cx, cy, fx, fy)

    if debug:
        return result, debug_info
    return result

def fallback_orientation(contour):
    """端点検出に失敗した場合のフォールバック（元のアルゴリズム）"""
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
title = "Control"
image_pairs = [
    {'original': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image.png', 
     'mask': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image_mask.tif'},
    {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h.png', 
     'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h_mask.tif'},
    {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h.png', 
     'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h_mask.tif'},
]

# title = "Vang KO in Glia"
# image_pairs = [
#     {'original': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001.png', 
#      'mask': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001_mask.tif'},
#     {'original': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.png', 
#      'mask': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002_mask.tif'},
#     {'original': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.png', 
#      'mask': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9_mask.tif'},
#     {'original': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.png', 
#      'mask': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001_mask.tif'},
# ]

# title = "Control (29℃)"
# image_pairs = [
#     {'original': './pics/29d_R8ori/Projections of 20251014_29degree_21hours_loco-Cas9-vang_R8_mCherry.png', 
#      'mask': './pics/29d_R8ori/Projections of 20251014_29degree_21hours_loco-Cas9-vang_R8_mCherry_mask.tif'},
# ]

# title = "Vang KO in Glia (29℃)"
# image_pairs = [
#     {'original': './pics/29d_R8control/Projections of 20251022_29degree_20h_senslexAlexOPmCherry_Cas9_no_sgRNA.png', 
#      'mask': './pics/29d_R8control/Projections of 20251022_29degree_20h_senslexAlexOPmCherry_Cas9_no_sgRNA_mask.tif'},
# ]

# image_pairs = [
#     {'original': './pics/mosaic_Fz2_control_fz2_Fz2/Projections of 20250413_24%APF_Fz2-GFP_fz2-RNAi_001_ue.tif', 
#      'mask': './pics/mosaic_Fz2_control_fz2_Fz2/Projections of 20250413_24%APF_Fz2-GFP_fz2-RNAi_001_ue_merge_mask.png'},
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

            # デバッグモードで関数を呼び出し
            if DEBUG_SKELETON:
                orientation_result, debug_info = calculate_horseshoe_orientation(contour, debug=True)
            else:
                orientation_result = calculate_horseshoe_orientation(contour, debug=False)
                debug_info = None

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
                # デバッグ: 骨格化と端点の表示
                # ----------------------------------------------------
                if DEBUG_SKELETON and debug_info:
                    # 骨格画像を拡大表示
                    skeleton_vis = debug_info['skeleton_vis'].copy()

                    # 選択された端点をシアンでハイライト
                    if 'selected_endpoints' in debug_info:
                        sel_ep1, sel_ep2 = debug_info['selected_endpoints']
                        cv2.circle(skeleton_vis, sel_ep1, 5, (255, 255, 0), 2)  # シアン枠
                        cv2.circle(skeleton_vis, sel_ep2, 5, (255, 255, 0), 2)  # シアン枠

                    display_size = (150, 150)
                    skeleton_resized = cv2.resize(skeleton_vis, display_size, interpolation=cv2.INTER_NEAREST)

                    # 情報テキストを追加
                    info_text = f"EP:{debug_info['num_endpoints']} BR:{debug_info['num_branch_points']}"
                    method = debug_info.get('selection_method', '')
                    method_short = "LP" if "longest_path" in method else ("D" if method == "direct" else "FB")
                    cv2.putText(skeleton_resized, info_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(skeleton_resized, method_short, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    # 馬蹄形ごとにウィンドウを表示
                    window_name = f"Skeleton #{valid_horseshoe_count_left + valid_horseshoe_count_right} ({side_label}) Angle:{angle:.1f}"
                    cv2.imshow(window_name, skeleton_resized)

                    # コンソールにも情報を出力
                    print(f"  [{side_label}] EP:{debug_info['num_endpoints']} BR:{debug_info['num_branch_points']} 方法:{debug_info.get('selection_method', 'N/A')} 角度:{angle:.1f}°") 

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
                
                if occupancy_rate > 0.30 or iterations >= 4:
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

    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(debug_img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

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

# 左側の極座標ヒストグラム（正規化: probability）
ax_left = plt.subplot(1, num_plots, 2, projection='polar')
if left_horseshoe_angles:
    left_angles_rad = np.radians(left_horseshoe_angles)
    angle_range = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False)
    counts, bin_edges = np.histogram(left_angles_rad, bins=angle_range)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)

    # 正規化: count → probability
    prob_left = counts / counts.sum() if counts.sum() > 0 else counts

    bars = ax_left.bar(centers, prob_left, width=width, color='magenta', alpha=0.6, bottom=0)

    ax_left.set_theta_zero_location("S")
    # r軸の最大値は後で左右の最大値から決定

    ax_left.set_xticks(np.radians([0, 90, 180, 270]))
    ax_left.set_xticklabels(['0°', '90°', '180°', ' -90°'])
    ax_left.set_rlabel_position(135)
    ax_left.set_title(f'Ventral (n={len(left_horseshoe_angles)})', va='bottom')
else:
    prob_left = np.array([0])
    ax_left.text(0, 0, 'No data', ha='center', va='center', transform=ax_left.transAxes)

# 右側の極座標ヒストグラム（正規化: probability）
ax_right = plt.subplot(1, num_plots, 3, projection='polar')
if right_horseshoe_angles:
    right_angles_rad = np.radians(right_horseshoe_angles)
    angle_range = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False)
    counts, bin_edges = np.histogram(right_angles_rad, bins=angle_range)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)

    # 正規化: count → probability
    prob_right = counts / counts.sum() if counts.sum() > 0 else counts

    bars = ax_right.bar(centers, prob_right, width=width, color='blue', alpha=0.6, bottom=0)

    ax_right.set_theta_zero_location("S")
    # r軸の最大値は後で左右の最大値から決定

    ax_right.set_xticks(np.radians([0, 90, 180, 270]))
    ax_right.set_xticklabels(['0°', '90°', '180°', ' -90°'])
    ax_right.set_rlabel_position(135)
    ax_right.set_title(f'Dorsal (n={len(right_horseshoe_angles)})', va='bottom')
else:
    prob_right = np.array([0])
    ax_right.text(0, 0, 'No data', ha='center', va='center', transform=ax_right.transAxes)

# 左右のr軸を統一（最大確率に基づく）
max_prob = max(prob_left.max() if len(prob_left) > 0 else 0,
               prob_right.max() if len(prob_right) > 0 else 0)
r_max = max_prob * 1.1 if max_prob > 0 else 1  # 少し余裕を持たせる
ax_left.set_rlim(0, r_max)
ax_right.set_rlim(0, r_max)

plt.tight_layout()
plt.show()