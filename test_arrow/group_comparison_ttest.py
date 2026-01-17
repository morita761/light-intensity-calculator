"""
Group Comparison T-Test Analysis for Horseshoe Orientation

This script compares horseshoe orientation between control and test groups,
performing t-tests on ventral and dorsal sides separately.

Usage:
    python group_comparison_ttest.py

Configure the CONTROL_FILES and TEST_FILES lists below with your image pairs.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats
from skimage.morphology import skeletonize
from collections import deque

# =============================================================================
# Configuration
# =============================================================================

# 除外する中心線の幅を定義
EXCLUSION_WIDTH = 80

# 極座標ヒストグラムのビンの幅 (-0.5° ~ 0.5°にするため)
BIN_WIDTH_DEG = 10.0        # -5°〜+5°
BIN_NUM = int(360 / BIN_WIDTH_DEG)+1  # 180

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

# =============================================================================
# Control群とTest群のファイルリスト
# =============================================================================

# Control群のファイルリスト
CONTROL_FILES = [
    {'original': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image.png',
     'mask': './pics/R8control/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_cas9_contorl.nd2...5degree_24hours_senslexAlexOP_loco_cas9_contorl_r_image_mask.tif'},
    {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h.png',
     'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_h_mask.tif'},
    {'original': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h.png',
     'mask': './pics/R8control/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi.nd2...5d_22h_senslexAlexOPmCherry_loco--cas9_x_sgRNAi_r_h_mask.tif'},
]

# CONTROL_FILES = [
#     {'original': './pics/29d_R8control/Projections of 20251022_29degree_20h_senslexAlexOPmCherry_Cas9_no_sgRNA.png', 
#      'mask': './pics/29d_R8control/Projections of 20251022_29degree_20h_senslexAlexOPmCherry_Cas9_no_sgRNA_mask.tif'},
# ]
# Test群のファイルリスト
TEST_FILES = [
    {'original': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001.png',
     'mask': './pics/R8ori/Projections of 20251105_25degree_24hours_senslexAlexOP_loco_vang_cas9001.nd2...5degree_24hours_senslexAlexOP_loco_vang_cas9001_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.png',
     'mask': './pics/R8ori/Projections of 20251119_25d_22h_senslexAlexOPmCherry_loco--vang-cas9002.nd2-...25d_22h_senslexAlexOPmCherry_loco--vang-cas9002_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.png',
     'mask': './pics/R8ori/Projections of 20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9.nd2-20251129_25d_22h_senslexAlexOPmCherry_loco_vang_cas9_mask.tif'},
    {'original': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.png',
     'mask': './pics/R8ori/Projections of 20251129_25d_22h_test_sensmCherry_loco_vang_cas9001.nd2-20251129_25d_22h_test_sensmCherry_loco_vang_cas9001_mask.tif'},
]
# TEST_FILES = [
#     {'original': './pics/29d_R8ori/Projections of 20251014_29degree_21hours_loco-Cas9-vang_R8_mCherry.png', 
#      'mask': './pics/29d_R8ori/Projections of 20251014_29degree_21hours_loco-Cas9-vang_R8_mCherry_mask.tif'},
# ]

# =============================================================================
# Helper Functions (from detect_arrow_improved.py)
# =============================================================================

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
    """画像全体の中心X座標を左右分割線として返す"""
    height, width = image.shape[:2]
    image_center_x = width // 2
    return image_center_x


def find_skeleton_endpoints_and_branches(skeleton):
    """骨格画像から端点と分岐点を検出する"""
    endpoints = []
    branch_points = []
    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - 1
                if neighbors == 1:
                    endpoints.append((j, i))
                elif neighbors >= 3:
                    branch_points.append((j, i))

    return endpoints, branch_points


def bfs_distance(skeleton, start, end):
    """骨格上でstartからendまでのBFS最短経路長を計算する"""
    rows, cols = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([(start[0], start[1], 0)])
    visited[start[1], start[0]] = True

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

    return -1


def find_longest_path_endpoints(skeleton, endpoints):
    """骨格グラフ上で最長パスを持つ端点ペアを見つける"""
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


def calculate_horseshoe_orientation(contour):
    """
    馬蹄形輪郭から凹（開口部）の方向を推定し、画像下=0度、右=90度、左=-90度の角度を返す。
    """
    if len(contour) < 5:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    padding = 5
    mask_w = w + 2 * padding
    mask_h = h + 2 * padding

    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    shifted_contour = contour - np.array([x - padding, y - padding])
    cv2.drawContours(mask, [shifted_contour], -1, 255, cv2.FILLED)

    skeleton = skeletonize(mask > 0)
    endpoints, branch_points = find_skeleton_endpoints_and_branches(skeleton)

    if len(endpoints) < 2:
        return fallback_orientation(contour)

    use_longest_path = len(endpoints) != 2 or len(branch_points) > 0

    if not use_longest_path:
        ep1, ep2 = endpoints[0], endpoints[1]
    else:
        result = find_longest_path_endpoints(skeleton, endpoints)
        if result:
            (ep1, ep2), path_dist = result
        else:
            max_dist = 0
            ep1, ep2 = endpoints[0], endpoints[1]
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 +
                                   (endpoints[i][1] - endpoints[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        ep1, ep2 = endpoints[i], endpoints[j]

    midpoint_local_x = (ep1[0] + ep2[0]) / 2
    midpoint_local_y = (ep1[1] + ep2[1]) / 2

    midpoint_x = midpoint_local_x + x - padding
    midpoint_y = midpoint_local_y + y - padding

    hull_points = cv2.convexHull(contour, returnPoints=True)
    M_hull = cv2.moments(hull_points)
    if M_hull["m00"] == 0:
        return None
    hull_cx = int(M_hull["m10"] / M_hull["m00"])
    hull_cy = int(M_hull["m01"] / M_hull["m00"])

    dx = midpoint_x - hull_cx
    dy = midpoint_y - hull_cy

    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return None

    dx_norm = dx / norm
    dy_norm = dy / norm

    cx = int(midpoint_x)
    cy = int(midpoint_y)

    arrow_length = max(w, h) * 0.6
    fx = int(cx + dx_norm * arrow_length)
    fy = int(cy + dy_norm * arrow_length)

    angle_rad_standard = np.arctan2(-dy, dx)
    angle_deg_standard = np.degrees(angle_rad_standard)

    orientation_angle = angle_deg_standard + 90

    if orientation_angle > 180:
        orientation_angle -= 360
    elif orientation_angle <= -180:
        orientation_angle += 360

    return (orientation_angle, cx, cy, fx, fy)


def fallback_orientation(contour):
    """端点検出に失敗した場合のフォールバック"""
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


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_image_pair(original_file, mask_file):
    """
    1つの画像ペアを処理し、左側（ventral）と右側（dorsal）の角度リストを返す

    Returns:
        tuple: (ventral_angles, dorsal_angles)
    """
    ventral_angles = []
    dorsal_angles = []

    image_original = cv2.imread(original_file)
    if image_original is None:
        print(f"  [Warning] Cannot read original image: {original_file}")
        return ventral_angles, dorsal_angles

    image_mask = cv2.imread(mask_file)
    if image_mask is None:
        print(f"  [Warning] Cannot read mask image: {mask_file}")
        return ventral_angles, dorsal_angles

    image_center_x = split_left_right(image_mask)

    all_horseshoe_masks = []
    for color_name in ANNOTATION_COLORS_HSV.keys():
        horseshoe_mask = extract_color_mask(image_mask, color_name)
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
            if is_overlapping:
                continue

            if contour_center_x + EXCLUSION_WIDTH < image_center_x:
                # 左側 (Ventral)
                orientation_result = calculate_horseshoe_orientation(contour)
                if orientation_result:
                    angle, _, _, _, _ = orientation_result
                    ventral_angles.append(angle)
            elif contour_center_x - EXCLUSION_WIDTH > image_center_x:
                # 右側 (Dorsal)
                orientation_result = calculate_horseshoe_orientation(contour)
                if orientation_result:
                    angle, _, _, _, _ = orientation_result
                    dorsal_angles.append(angle)

    return ventral_angles, dorsal_angles


def process_group(file_list, group_name):
    """
    グループ内の全ファイルを処理し、ventral/dorsalの角度を集計

    Returns:
        dict: {'ventral': [...], 'dorsal': [...]}
    """
    all_ventral = []
    all_dorsal = []

    print(f"\n{'='*60}")
    print(f"Processing {group_name} group ({len(file_list)} files)")
    print('='*60)

    for i, file_info in enumerate(file_list):
        original_file = file_info['original']
        mask_file = file_info['mask']

        print(f"\n  [{i+1}/{len(file_list)}] {os.path.basename(original_file)}")

        ventral, dorsal = process_image_pair(original_file, mask_file)
        all_ventral.extend(ventral)
        all_dorsal.extend(dorsal)

        print(f"    Ventral: {len(ventral)} cells, Dorsal: {len(dorsal)} cells")

    print(f"\n  {group_name} Total: Ventral={len(all_ventral)}, Dorsal={len(all_dorsal)}")

    return {'ventral': all_ventral, 'dorsal': all_dorsal}


def perform_ttest(control_data, test_data, side_name):
    """
    2群間のt検定を実行

    Args:
        control_data: Control群の角度リスト
        test_data: Test群の角度リスト
        side_name: "Ventral" or "Dorsal"

    Returns:
        dict: t検定の結果
    """
    result = {
        'side': side_name,
        'control_n': len(control_data),
        'test_n': len(test_data),
        'control_mean': np.nan,
        'control_std': np.nan,
        'test_mean': np.nan,
        'test_std': np.nan,
        't_statistic': np.nan,
        'p_value': np.nan
    }

    if len(control_data) < 2 or len(test_data) < 2:
        print(f"  [Warning] Not enough data for t-test ({side_name})")
        print(f"    Control n={len(control_data)}, Test n={len(test_data)}")
        return result

    result['control_mean'] = np.mean(control_data)
    result['control_std'] = np.std(control_data, ddof=1)
    result['test_mean'] = np.mean(test_data)
    result['test_std'] = np.std(test_data, ddof=1)

    # Welch's t-test (equal_var=False)
    t_stat, p_value = stats.ttest_ind(control_data, test_data, equal_var=False)

    result['t_statistic'] = t_stat
    result['p_value'] = p_value

    return result


def print_ttest_results(ventral_result, dorsal_result):
    """t検定の結果を表示"""
    print("\n" + "="*70)
    print("T-TEST RESULTS")
    print("="*70)

    for result in [ventral_result, dorsal_result]:
        print(f"\n{result['side']} Side:")
        print("-"*50)
        print(f"  Control: n={result['control_n']}, "
              f"mean={result['control_mean']:.2f}°, "
              f"std={result['control_std']:.2f}°")
        print(f"  vang Cas9:    n={result['test_n']}, "
              f"mean={result['test_mean']:.2f}°, "
              f"std={result['test_std']:.2f}°")
        print(f"  t-statistic = {result['t_statistic']:.4f}")
        print(f"  p-value = {result['p_value']:.10f}", end="")

        if result['p_value'] < 0.001:
            print(" ***")
        elif result['p_value'] < 0.01:
            print(" **")
        elif result['p_value'] < 0.05:
            print(" *")
        else:
            print(" (n.s.)")


def get_asterisks(p_value):
    """p値からアスタリスクを返す"""
    if np.isnan(p_value):
        return 'n.s.'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'


def plot_results(control_data, test_data, ventral_result, dorsal_result):
    """結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # =================================
    # 1. Ventral Box Plot (上段左)
    # =================================
    ax1 = axes[0, 0]
    data_ventral = [control_data['ventral'], test_data['ventral']]
    bp1 = ax1.boxplot(data_ventral, labels=['Control', 'vang Cas9'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')

    ax1.set_ylabel('Orientation Angle (°)')
    ax1.set_title(f"Ventral Side\n"
                  f"Control: n={ventral_result['control_n']}, "
                  f"mean={ventral_result['control_mean']:.1f}° ± {ventral_result['control_std']:.1f}°\n"
                  f"vang Cas9: n={ventral_result['test_n']}, "
                  f"mean={ventral_result['test_mean']:.1f}° ± {ventral_result['test_std']:.1f}°")
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(-180, 220)

    # 線とアスタリスクの描画 (Ventral)
    asterisks_ventral = get_asterisks(ventral_result['p_value'])
    line_y1 = 120
    star_y1 = line_y1 + 3
    ax1.plot([1, 1, 2, 2], [line_y1, line_y1 + 5, line_y1 + 5, line_y1], lw=1.5, c='black')
    ax1.text(1.5, star_y1, asterisks_ventral, ha='center', va='bottom', fontsize=14)

    # =================================
    # 2. Dorsal Box Plot (上段右)
    # =================================
    ax2 = axes[0, 1]
    data_dorsal = [control_data['dorsal'], test_data['dorsal']]
    bp2 = ax2.boxplot(data_dorsal, labels=['Control', 'vang Cas9'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')

    ax2.set_ylabel('Orientation Angle (°)')
    ax2.set_title(f"Dorsal Side\n"
                  f"Control: n={dorsal_result['control_n']}, "
                  f"mean={dorsal_result['control_mean']:.1f}° ± {dorsal_result['control_std']:.1f}°\n"
                  f"vang Cas9: n={dorsal_result['test_n']}, "
                  f"mean={dorsal_result['test_mean']:.1f}° ± {dorsal_result['test_std']:.1f}°")
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(-180, 220)

    # 線とアスタリスクの描画 (Dorsal)
    asterisks_dorsal = get_asterisks(dorsal_result['p_value'])
    line_y2 = 120
    star_y2 = line_y2 + 3
    ax2.plot([1, 1, 2, 2], [line_y2, line_y2 + 5, line_y2 + 5, line_y2], lw=1.5, c='black')
    ax2.text(1.5, star_y2, asterisks_dorsal, ha='center', va='bottom', fontsize=14)

    # =================================
    # 3. Ventral Polar Histogram (下段左)
    # =================================
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    bin_edges = np.linspace(-np.pi, np.pi, BIN_NUM, endpoint=False)

    if control_data['ventral']:
        ctrl_rad = np.radians(control_data['ventral'])
        counts_ctrl, _ = np.histogram(ctrl_rad, bins=bin_edges)
        prob_ctrl = counts_ctrl / counts_ctrl.sum() if counts_ctrl.sum() > 0 else counts_ctrl
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)
        ax3.bar(centers, prob_ctrl, width=width, color='blue', alpha=0.5, label='Control')

    if test_data['ventral']:
        test_rad = np.radians(test_data['ventral'])
        counts_test, _ = np.histogram(test_rad, bins=bin_edges)
        prob_test_l = counts_test / counts_test.sum() if counts_test.sum() > 0 else counts_test
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)
        ax3.bar(centers, prob_test_l, width=width, color='red', alpha=0.5, label='vang Cas9')

    ax3.set_theta_zero_location("S")
    ax3.set_xticks(np.radians([0, 90, 180, 270]))
    ax3.set_xticklabels(['0°', '90°', '180°', '-90°'])
    ax3.set_rlabel_position(135)
    ax3.set_title('Ventral - Polar Histogram', va='bottom', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # =================================
    # 4. Dorsal Polar Histogram (下段右)
    # =================================
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')

    if control_data['dorsal']:
        ctrl_rad = np.radians(control_data['dorsal'])
        counts_ctrl, _ = np.histogram(ctrl_rad, bins=bin_edges)
        prob_ctrl = counts_ctrl / counts_ctrl.sum() if counts_ctrl.sum() > 0 else counts_ctrl
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)
        ax4.bar(centers, prob_ctrl, width=width, color='blue', alpha=0.5, label='Control')

    if test_data['dorsal']:
        test_rad = np.radians(test_data['dorsal'])
        counts_test, _ = np.histogram(test_rad, bins=bin_edges)
        prob_test_r = counts_test / counts_test.sum() if counts_test.sum() > 0 else counts_test
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)
        ax4.bar(centers, prob_test_r, width=width, color='red', alpha=0.5, label='vang Cas9')

    ax4.set_theta_zero_location("S")
    ax4.set_xticks(np.radians([0, 90, 180, 270]))
    ax4.set_xticklabels(['0°', '90°', '180°', '-90°'])
    ax4.set_rlabel_position(135)
    ax4.set_title('Dorsal - Polar Histogram', va='bottom', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 左右のr軸を統一（最大確率に基づく）
    max_prob = max(prob_test_l.max() if len(prob_test_l) > 0 else 0,
                   prob_test_r.max() if len(prob_test_r) > 0 else 0)
    r_max = max_prob * 1.1 if max_prob > 0 else 1  # 少し余裕を持たせる
    ax3.set_rlim(0, r_max)
    ax4.set_rlim(0, r_max)

    # Remove the default subplot created for polar plots
    axes[1, 0].remove()
    axes[1, 1].remove()

    plt.tight_layout()
    plt.savefig('group_comparison_results.png', dpi=150, bbox_inches='tight')
    print("\n[Info] Results saved to 'group_comparison_results.png'")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("Group Comparison T-Test Analysis for Horseshoe Orientation")
    print("="*70)

    # Control群の処理
    control_data = process_group(CONTROL_FILES, "Control")

    # Test群の処理
    test_data = process_group(TEST_FILES, "vang Cas9")

    # T検定の実行
    print("\n" + "="*70)
    print("Performing T-Tests...")
    print("="*70)

    ventral_result = perform_ttest(
        control_data['ventral'],
        test_data['ventral'],
        "Ventral"
    )

    dorsal_result = perform_ttest(
        control_data['dorsal'],
        test_data['dorsal'],
        "Dorsal"
    )

    # 結果の表示
    print_ttest_results(ventral_result, dorsal_result)

    # 可視化
    plot_results(control_data, test_data, ventral_result, dorsal_result)

    return control_data, test_data, ventral_result, dorsal_result


if __name__ == "__main__":
    main()
