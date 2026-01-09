import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import sys

# 定数設定
# 除外する中心線の幅を定義 (ここでは使っていませんが、元のコードに合わせて保持)
EXCLUSION_WIDTH = 50  # 中心線からの無視する距離
DEBUG_FLAG = 0
target_size = (100, 100) 

# アノテーションに使用する色のHSV範囲を定義 (元のコードから変更なし)
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

# --- ユーティリティ関数（元のコードから変更なし） ---
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

image_pairs_cas9 = [
    {'original': './pics/vang/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue.png', 
     'mask': './pics/vang/Projections of 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita.png', 
     'mask': './pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_sita_mask.tif'},
    {'original':'./pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue.png', 
     'mask': './pics/vang/Projections of 20250413_24%APF_Fz-GFP_loco_vang_cas9P_001_ue_mask.tif'},
    {'original':'./pics/vang/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2.png', 
     'mask': './pics/vang/Projections of 20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P.nd2-20250518_25degree_24%APF_Fz-GFP_loco-vang-Cas9P_2_mask.tif'}
]

# 全ての画像ペアを結合（グローバル平均輝度計算のため）
all_image_pairs = image_pairs_control + image_pairs_cas9

# =========================================================================
# 【STEP 1】 全体の基準平均輝度 (global_mean_intensity) の計算
# =========================================================================
all_raw_gfp_values = [] 
debug_images_conttrol = []
debug_images_cas9 = []

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
print(f"✅ 全体の基準平均輝度 (Global Mean): {global_mean_intensity:.2f}")


# =========================================================================
# 【STEP 2】 データ収集処理の関数化
# =========================================================================

def collect_normalized_gfp_data(image_pairs, global_mean_intensity, target_size):
    """
    画像ペアのリストから、正規化・リサイズされた左右のGFP輝度データを収集する。
    
    Args:
        image_pairs (list): 処理対象の画像ペアのリスト。
        global_mean_intensity (float): 全体の平均輝度。
        target_size (tuple): リサイズ後のターゲットサイズ (幅, 高さ)。
        
    Returns:
        tuple: (all_aligned_gfp_data_left, all_aligned_gfp_data_right)
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
        
        # 現在の画像の平均輝度を計算 (global_mean_intensity の計算と同じロジック)
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
        # マスク画像ではなく、**オリジナル画像**をベースとして、その上にアノテーションの色を重ねる
        debug_img = image_original_for_gfp.copy()

        # 中心線を白で描画
        cv2.line(debug_img, (image_center_x, 0), (image_center_x, debug_img.shape[0]), (255, 255, 255), 1)
        all_horseshoe_masks = []
        for color_name in ANNOTATION_COLORS_HSV.keys():
            horseshoe_mask = extract_color_mask(image_mask_for_contours, color_name)
            all_horseshoe_masks.append(horseshoe_mask)

        # 各輪郭を左右に分類し、GFP輝度データを抽出
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
                    # 中心線に近すぎる、または被る輪郭は無視
                    continue 
                elif contour_center_x < image_center_x:
                    current_gfp_data_list = all_aligned_gfp_data_left
                    color = (0, 0, 255)
                else: # contour_center_x > image_center_x
                    current_gfp_data_list = all_aligned_gfp_data_right
                    color = (255, 0, 0)
                # 膨張処理用のローカルマスクの作成 (元のロジックを簡略化して適用)
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                rel_contour = contour - np.array([x, y])
                cv2.drawContours(temp_mask, [rel_contour], -1, 255, cv2.FILLED)
                
                # 膨張ロジック（簡略化: 膨張は行わないが、マスクは使用）
                contour_mask_local = temp_mask.copy()
                
                # GFPデータの切り出しとマスク適用
                # 最終的な輪郭を取得し、debug_imgに描画
                final_contours, _ = cv2.findContours(contour_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if final_contours:
                    final_absolute_contour = final_contours[0] + np.array([x, y])
                    # 【修正3: アノテーション画像の個別出力】検出した輪郭をdebug_img（オリジナル画像ベース）に赤/青で描画
                    cv2.drawContours(debug_img, [final_absolute_contour], -1, color, -1)
                gfp_roi = gfp_channel[y:y+h, x:x+w]
                masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)

                # リサイズ
                resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
                
                # 輝度正規化
                resized_gfp_float = resized_gfp.astype(np.float64) 
                resized_gfp_normalized_float = resized_gfp_float * scaling_factor
                
                # 結果を float64 のままリストに保存（平均化の際に精度を保つため）
                current_gfp_data_list.append(resized_gfp_normalized_float)

        debug_images.append(debug_img)
    return all_aligned_gfp_data_left, all_aligned_gfp_data_right, debug_images


# =========================================================================
# 【STEP 3】 controlとcas9のデータ収集
# =========================================================================

print("\n--- Controlデータ収集開始 ---")
control_left_data, control_right_data, debug_images_conttrol = collect_normalized_gfp_data(
    image_pairs_control, global_mean_intensity, target_size
)

print("--- Cas9データ収集開始 ---")
cas9_left_data, cas9_right_data, debug_images_cas9 = collect_normalized_gfp_data(
    image_pairs_cas9, global_mean_intensity, target_size
)

# =========================================================================
# 【STEP 4】 左右それぞれの平均ヒートマップの計算
#           ※ ここで初めて、すべてのデータに対して平均化を行う
# =========================================================================

def calculate_average_heatmap(data_list, label):
    if data_list:
        # float64で平均を計算
        avg_heatmap = np.mean(np.array(data_list), axis=0) 
        print(f"✅ {label} ({len(data_list)}個のデータ) の平均ヒートマップを計算しました。")
        return avg_heatmap
    else:
        print(f"❌ {label} のデータが不足しているため、平均ヒートマップはゼロで初期化されます。")
        return np.zeros(target_size, dtype=np.float64)

def normalize_average_heatmap(data_list, label, max):
    # avg_heatmap_n = cv2.normalize(data_list, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    avg_heatmap_n  = (data_list  / max * 255).astype(np.uint8)
    imshow = cv2.applyColorMap(avg_heatmap_n, cv2.COLORMAP_VIRIDIS)
    window_name = f"Average GFP Intensity Heatmap \n({label} Horseshoes)"
    # cv2.imshow(window_name, imshow)
    # cv2.waitKey(0) 
    
    # # 保存用のヒートマップの表示
    # fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # figsizeで図のサイズを指定できます
    # im = ax.imshow(avg_heatmap_n, cmap='viridis', vmin=0, vmax=255)
    # # カラーバーの追加
    # plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity')
    # # タイトルの設定
    # ax.set_title(window_name)
    # # 軸の非表示
    # ax.axis('off')
    # # 図の表示
    # plt.show()
    return avg_heatmap_n

def maxmax(list1, list2, list3, list4):
    max_1 = np.max(list1)
    max_2 = np.max(list2)
    max_3 = np.max(list3)
    max_4 = np.max(list4)
    maximam = max(max_1, max_2, max_3, max_4)
    print(f"max = { maximam}")
    return maximam

# controlの平均ヒートマップ
avg_control_left = calculate_average_heatmap(control_left_data, "Control Left")
avg_control_right = calculate_average_heatmap(control_right_data, "Control Right")

# cas9の平均ヒートマップ
avg_cas9_left = calculate_average_heatmap(cas9_left_data, "Cas9 Left")
avg_cas9_right = calculate_average_heatmap(cas9_right_data, "Cas9 Right")

max = maxmax(avg_control_left, avg_control_right, avg_cas9_left, avg_cas9_right)
# ヒートマップで描画
avg_control_left = normalize_average_heatmap(avg_control_left, "Left", max)
avg_control_right = normalize_average_heatmap(avg_control_right, "Right", max)
avg_cas9_left = normalize_average_heatmap(avg_cas9_left, "Left", max)
avg_cas9_right = normalize_average_heatmap(avg_cas9_right, "Right", max)

# # 保存用のヒートマップの表示
# fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # figsizeで図のサイズを指定できます
# im = ax.imshow(left_heatmap_colored, cmap='viridis', vmin=0, vmax=255)
# # カラーバーの追加
# plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity')
# # タイトルの設定
# ax.set_title('Average GFP Intensity Heatmap \n(Left Horseshoes)')
# # 軸の非表示
# ax.axis('off')
# # 図の表示
# plt.show()

# =========================================================================
# 【STEP 5】 左右それぞれの差分ヒートマップ (Control - Cas9) の計算と表示
# =========================================================================

# def plot_difference_heatmap(avg_control, avg_cas9, side_label):
#     # 差分ヒートマップの計算
#     difference_heatmap = avg_control - avg_cas9
    
#     # 全体の最小値・最大値を計算し、カラーバーの中心を0にする
#     max_abs_diff = np.nanmax(np.abs(difference_heatmap)) # nanを考慮
    
#     plt.figure(figsize=(6, 5))
#     # cmap='seismic'で、中心が白、両端が赤/青になるように設定
#     # vminとvmaxを-max_abs_diffから+max_abs_diffに設定することで、カラーバーの中心を0に固定
#     sns.heatmap(
#         difference_heatmap, 
#         cmap='seismic', 
#         center=0, 
#         vmin=-200, 
#         vmax=200,
#         cbar_kws={'label': 'Difference in Normalized GFP Intensity (Control - Cas9)'}
#     )
#     plt.title(f'Difference Heatmap (Control - Cas9) - {side_label}', fontsize=14)
#     plt.xlabel(f'X-axis ({target_size[0]} pixels)')
#     plt.ylabel(f'Y-axis ({target_size[1]} pixels)')
#     plt.gca().invert_yaxis() # 画像のY軸に合わせて反転
#     plt.show()

# # 左側（Ventral）の差分ヒートマップを表示
# plot_difference_heatmap(avg_control_left, avg_cas9_left, "Left (Ventral)")

# # 右側（Dorsal）の差分ヒートマップを表示
# plot_difference_heatmap(avg_control_right, avg_cas9_right, "Right (Dorsal)")

#=========================================================================================
# --- 1. 左右それぞれの差分ヒートマップを計算 ---
difference_left = avg_cas9_left - avg_control_left
difference_right = avg_cas9_right - avg_control_right

# --- 2. 左右全体で最も大きい絶対差を計算 ---
# nanを無視して、左右の差分データ全体から最大絶対値を求める
max_abs_diff_left = np.nanmin(np.abs(difference_left))
max_abs_diff_right = np.nanmax(np.abs(difference_right))
# 左右どちらかの最大値（より大きな値）を全体の軸の最大値として採用
# 修正案 1: .item()でPythonのfloatに変換してから比較
# global_max_abs_diff = max(max_abs_diff_left.item(), max_abs_diff_right.item())
# 修正案 2: NumPyの関数を使用
global_max_abs_diff = np.maximum(max_abs_diff_left, max_abs_diff_right)
# 結果はnp.float64型で返されます
# global_max_abs_diff = 255

# 軸の範囲を少し広げて、端のデータが見切れないようにする（任意）
# ここでは丸め処理は行わず、そのまま使用します。
# 視覚的な調整が必要であれば、 global_max_abs_diff * 1.05 などとしても良い。
axis_limit = global_max_abs_diff

print(f"左右全体での最大絶対差分: {axis_limit:.2f}")

# --- 3. plot_difference_heatmap関数を修正し、軸の情報を引数として受け取るようにする ---
def plot_difference_heatmap(difference_data, side_label, vmin_val, vmax_val):
    avg_heatmap_n = cv2.normalize(difference_data, None, -50, 50, cv2.NORM_MINMAX).astype(np.uint8)
    # avg_heatmap_n = difference_data
    # avg_heatmap_n  = (data_list  / max * 255).astype(np.uint8)
    imshow = cv2.applyColorMap(avg_heatmap_n, cv2.COLORMAP_VIRIDIS)
    window_name = f"Average GFP Intensity Heatmap \n({side_label} Horseshoes)"
    cv2.imshow(window_name, imshow)
    cv2.waitKey(0) 

    # 保存用のヒートマップの表示
    fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # figsizeで図のサイズを指定できます
    im = ax.imshow(avg_heatmap_n, cmap='viridis', vmin=-50, vmax=50)
    # カラーバーの追加
    plt.colorbar(im, ax=ax, label='Normalized Average GFP Intensity')
    # タイトルの設定
    ax.set_title(window_name)
    # 軸の非表示
    ax.axis('off')
    # 図の表示
    plt.show()

    # plt.figure(figsize=(6, 5))
    
    # sns.heatmap(
    #     difference_data, 
    #     cmap='seismic', 
    #     center=0, 
    #     vmin=vmin_val,        # 共通の軸範囲を使用
    #     vmax=vmax_val,        # 共通の軸範囲を使用
    #     cbar_kws={'label': 'Difference in Normalized GFP Intensity (Cas9 - Control)'}
    # )
    # plt.title(f'Difference Heatmap (Cas9 - Control) - {side_label}', fontsize=14)
    # plt.xlabel(f'X-axis ({difference_data.shape[1]} pixels)')
    # plt.ylabel(f'Y-axis ({difference_data.shape[0]} pixels)')
    # plt.gca().invert_yaxis()
    # plt.show()

# --- 4. 共通の軸範囲を使って関数を呼び出す ---
# plot_difference_heatmap(difference_left, "Left (Ventral)", -axis_limit, axis_limit)

# plot_difference_heatmap(difference_right, "Right (Dorsal)", -axis_limit, axis_limit)

# Matplotlibで統合プロットを作成 (視覚的な確認用)
num_plots = 3
num_image_sets = len(image_pairs_control) 
num_rows_for_display = num_image_sets if num_image_sets > 0 else 1 

fig = plt.figure(figsize=(num_plots * 4, 4))

plot_index = 1

rows_inner = 2
cols_inner = 2
num_debug_images = len(debug_images_conttrol)

# 表示用
imshow_image = debug_images_conttrol
imshow_left = avg_control_left
imshow_right = avg_control_right

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
            ax.imshow(cv2.cvtColor(imshow_image[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f'Annotated {i+1}', fontsize=10) # タイトルを小さくする
            ax.axis('off')
else:
    # Debug画像のプロット
    for i in range(num_image_sets):
        plt.subplot(num_rows_for_display, num_plots, i * num_plots + 1)
        plt.imshow(cv2.cvtColor(imshow_image[i], cv2.COLOR_BGR2RGB))
        plt.title(f'Annotated Image {i+1}')
        plt.axis('off')
plot_index += 1

# 左側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(imshow_left, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Ventral Horseshoes)')
plt.axis('off')
plot_index += 1

# 右側のヒートマップのプロット
plt.subplot(1, num_plots, plot_index)
plt.imshow(imshow_right, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap \n(Dorsal Horseshoes)')
plt.axis('off')
plot_index += 1

plt.tight_layout()
plt.show()

# ヒストグラムを表示
def histgram(average_heatmap_left_normalized, average_heatmap_right_normalized):
    data_left_flat = np.concatenate([arr.flatten() for arr in average_heatmap_left_normalized])
    data_right_flat = np.concatenate([arr.flatten() for arr in average_heatmap_right_normalized])
    max1 = np.max(average_heatmap_left_normalized)
    max2 = np.max(average_heatmap_right_normalized)
    if(max1 > max2):
        max = max1
    else:
        max = max2

    plt.figure(figsize=(10, 5))

    # 左側のヒストグラム
    plt.subplot(1, 2, 1)
    # bins の設定や range の設定は、データに合わせて調整してください。
    plt.hist(data_left_flat, bins=50, range=(0, max), color='red', alpha=0.6, label='Left Horseshoes')
    # plt.title('Normalized GFP Intensity Distribution (Left)')
    plt.title('Normalized GFP Intensity Distribution (Cas9)')
    # plt.xlabel('Intensity Value (0-255)')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)

    # 右側のヒストグラム
    plt.subplot(1, 2, 2)
    plt.hist(data_right_flat, bins=50, range=(0, max), color='blue', alpha=0.6, label='Right Horseshoes')
    # plt.title('Normalized GFP Intensity Distribution (Right)')
    plt.title('Normalized GFP Intensity Distribution (control)')
    # plt.xlabel('Intensity Value (0-255)')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.show()

# histgram(avg_control_left, avg_control_right)
# histgram(avg_cas9_right, avg_control_right)
histgram(avg_cas9_left, avg_control_left)

# デバッグ画像表示用のウィンドウを閉じる (元のコードのまま)
cv2.waitKey(1) 
cv2.destroyAllWindows()