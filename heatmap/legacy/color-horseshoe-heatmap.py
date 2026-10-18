import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# アノテーションに使用する色のHSV範囲を定義
# 各色のHue（色相）範囲を厳密に設定することが重要
# Saturation (彩度) と Value (明度) は広めに取るのが一般的ですが、ノイズが入る場合は調整
ANNOTATION_COLORS_HSV = {
    "red": {
        "lower": np.array([0, 100, 100]),   # Hueが0-10, 170-180の間
        "upper": np.array([10, 255, 255]),
        "lower2": np.array([170, 100, 100]), # 赤はHueが両端にあるため2つの範囲が必要
        "upper2": np.array([180, 255, 255])
    },
    "blue": {
        "lower": np.array([100, 100, 100]), # Hueが100-130くらい
        "upper": np.array([130, 255, 255])
    },
    "cyan": {
        "lower": np.array([85, 100, 100]), # Hueが85-95くらい
        "upper": np.array([100, 255, 255])
    },
    "magenta": {
        "lower": np.array([140, 100, 100]), # Hueが140-160くらい
        "upper": np.array([170, 255, 255])
    }
    # 他の色を追加する場合はここに追加
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

    # ノイズ除去と形状の連結 (オプションだが推奨)
    # カーネルサイズやイテレーション数はデータに合わせて調整
    # 線が途切れるのでなし
    # kernel = np.ones((3,3), np.uint8)
    # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1) # 小さなノイズ除去
    # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1) # ギャップの連結

    return final_mask

def extract_region(image, coefficient_top=0.2,
                   coefficient_bottom=0.8,
                   coefficient_left=0.25,
                   coefficient_right=0.66):
    """
    元画像から指定された割合の領域を切り抜きます。
    この関数は緑色抽出を行わず、純粋な画像領域の切り抜きに特化しています。

    Args:
        image (np.array): 入力画像 (BGR形式)。
        coefficient_top (float): 上部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_bottom (float): 下部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_left (float): 左部から切り抜く領域の割合 (0.0 - 1.0)。
        coefficient_right (float): 右部から切り抜く領域の割合 (0.0 - 1.0)。

    Returns:
        np.array: 指定された領域が切り抜かれた画像。
                  指定領域外は黒（0）になります。
    """
    height, width = image.shape[:2]

    # 切り抜き範囲を計算
    y_start = int(height * coefficient_top)
    y_end = int(height * coefficient_bottom)
    x_start = int(width * coefficient_left)
    x_end = int(width * coefficient_right)

    # 確実に有効なインデックス範囲内に収める
    y_start = max(0, min(y_start, height - 1))
    y_end = max(0, min(y_end, height))
    x_start = max(0, min(x_start, width - 1))
    x_end = max(0, min(x_end, width))

    # 切り抜き領域のマスクを作成 (指定領域内が白、外が黒)
    clip_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    clip_mask[y_start:y_end, x_start:x_end] = 255

    # 元画像にマスクを適用して領域を切り抜く
    clipped_image = cv2.bitwise_and(image, image, mask=clip_mask)

    # 余白をなくして純粋に切り抜かれた矩形領域を返す
    return clipped_image[y_start:y_end, x_start:x_end].copy()


def split_left_right(image):
    """
    入力画像内の緑色領域の垂直方向の中心線X座標を返します。
    このX座標は、後で馬蹄形輪郭を左右に分類するために使用されます。

    Args:
        image (np.array): 緑色領域を含む画像（BGR形式）。

    Returns:
        int: 緑色領域の垂直方向の中心線X座標。
             緑色領域が見つからない場合は、画像全体の中心X座標を返します。
    """
    height, width = image.shape[:2]

    # BGR → HSV に変換して緑色領域を検出
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])  # 緑色のHSV下限値（調整可能）
    upper_green = np.array([80, 255, 255]) # 緑色のHSV上限値（調整可能）
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 緑色領域のバウンディングボックスを計算
    if cv2.countNonZero(green_mask) > 0:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(green_mask))
    else:
        print("警告: 画像内に緑色領域が見つかりませんでした。画像全体の中心で分割します。")
        x, y, w, h = 0, 0, width, height # 画像全体を対象とする

    # 緑色領域の中心X座標を計算
    green_center_x = x + w // 2

    return green_center_x

# --- メイン処理 ---
# 1. 複数の画像のファイル名を指定
# image_files = ['002_stronger_mask.tif', '003_stronger_mask.tif'] # ここに処理したい画像ファイル名を追加
image_files = ['Fz-RFP_001_green_mask.tif']

all_aligned_gfp_data_left = []  # 左側の馬蹄形から得られるGFPデータを格納するリスト
all_aligned_gfp_data_right = [] # 右側の馬蹄形から得られるGFPデータを格納するリスト
processed_images_for_plot = []  # プロット用の切り抜かれたオリジナル画像を格納するリスト
debug_images = []               # debug表示用の画像を格納するリスト

target_size = (100, 100) # 例: 100x100ピクセルに統一 (ヒートマップのサイズ)

for file_name in image_files:
    # 1. 画像の読み込み
    image_original = cv2.imread(file_name)
    if image_original is None:
        print(f"画像を読み込めませんでした: {file_name}。パスを確認してください。")
        continue

    # 必要な領域を切り抜く
    green_mask_image = extract_region(image_original, coefficient_left=0.23, coefficient_right=0.75, coefficient_bottom=0.92, coefficient_top=0.09)
    processed_images_for_plot.append(cv2.cvtColor(green_mask_image.copy(), cv2.COLOR_BGR2RGB)) # Matplotlib用にBGRからRGBに変換
    
    # 緑色領域の中心X座標を取得
    green_center_x = split_left_right(green_mask_image)

    # debug表示用に画像をコピー
    debug_img = green_mask_image.copy()
    # 中心線を白で描画
    cv2.line(debug_img, (green_center_x, 0), (green_center_x, debug_img.shape[0]), (255, 255, 255), 1)
    cv2.imshow("sample1", debug_img)
    cv2.waitKey(0)

    all_horseshoe_masks = []

    # extract_color_maskを使って、馬蹄形輪郭を抽出
    for color_name in ANNOTATION_COLORS_HSV.keys():
        # 特定の色（インスタンス）のマスクを抽出
        horseshoe_mask = extract_color_mask(green_mask_image, color_name)
        all_horseshoe_masks.append(horseshoe_mask)
        cv2.imshow("sample1", horseshoe_mask)
        cv2.waitKey(0)   

    # 全てのマスクを結合
    combined_horseshoe_mask = all_horseshoe_masks[0]
    for i in range(1, len(all_horseshoe_masks)):
        combined_horseshoe_mask = cv2.bitwise_or(combined_horseshoe_mask, all_horseshoe_masks[i])
    print(len(combined_horseshoe_mask))

    # 各インスタンスマスクから輪郭を検出
    # RETR_EXTERNAL で外側の輪郭のみを検出（各色の塊ごとに）
    contours, _ = cv2.findContours(combined_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gfp_channel = green_mask_image[:, :, 1] # 緑色チャネルを抽出（GFP輝度）

    # 各輪郭を左右に分類し、GFP輝度データを抽出
    valid_horseshoe_count_left = 0
    valid_horseshoe_count_right = 0

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 20: # 小さすぎるノイズ輪郭を除外 (適宜調整)
            continue

        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)

        # 輪郭の中心X座標
        threashold = 40
        contour_center_x = x + w // 2

        # 中心線に被る輪郭を無視する閾値
        # 輪郭のバウンディングボックスが中心線をまたぐかをチェック
        # 例えば、輪郭の左端が中心線より右、かつ、輪郭の右端が中心線より左の場合
        is_overlapping = (x < green_center_x < x + w)

        if is_overlapping:
            # 中心線に被る輪郭は無視
            color = (0, 255, 255) # 黄色 (BGR) for ignored contours
            cv2.drawContours(debug_img, [contour], -1, color, -1) # debug用に描画したい場合
            continue # この輪郭はスキップ
        elif contour_center_x + threashold < green_center_x:
            # 左側の輪郭
            current_gfp_data_list = all_aligned_gfp_data_left
            color = (0, 0, 255) # 赤 (BGR) for debug
            valid_horseshoe_count_left += 1
        elif contour_center_x  - threashold > green_center_x:
            # 右側の輪郭
            current_gfp_data_list = all_aligned_gfp_data_right
            color = (255, 0, 0) # 青 (BGR) for debug
            valid_horseshoe_count_right += 1
        else:
            color = (0, 255, 255) # 黄色 (BGR) for ignored contours
            cv2.drawContours(debug_img, [contour], -1, color, -1) # debug用に描画したい場合

        # debug用に輪郭を描画
        cv2.drawContours(debug_img, [contour], -1, color, -1)

        # 馬蹄形内部の輝度を抽出するためのマスクを作成
        # まず、輪郭のバウンディングボックスサイズの黒い画像を作成
        contour_mask_local = np.zeros((h, w), dtype=np.uint8)
        # 輪郭をこのローカルマスクに描画 (相対座標に変換)
        rel_contour = contour - np.array([x, y])
        cv2.drawContours(contour_mask_local, [rel_contour], -1, 255, cv2.FILLED)

        # 元のGFPチャネル画像から、このバウンディングボックスに対応する部分を切り出す
        gfp_roi = gfp_channel[y:y+h, x:x+w]

        # 切り出したGFP輝度データに、作成したローカルマスクを適用
        masked_gfp_roi = cv2.bitwise_and(gfp_roi, gfp_roi, mask=contour_mask_local)

        # リサイズ (target_sizeに統一)
        resized_gfp = cv2.resize(masked_gfp_roi, target_size, interpolation=cv2.INTER_LINEAR)
        resized_gfp = np.clip(resized_gfp, 0, 255) # 0-255の範囲にクリップ

        current_gfp_data_list.append(resized_gfp)
    
    cv2.imshow("sample1", debug_img)
    cv2.waitKey(0)

    debug_images.append(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)) # Matplotlib用にBGRからRGBに変換

    print(f"{file_name} から左側の有効な馬蹄形が {valid_horseshoe_count_left} 個、右側の有効な馬蹄形が {valid_horseshoe_count_right} 個検出されました。")

# --- 4. すべての輝度データの平均化 ---
average_heatmap_left = None
if all_aligned_gfp_data_left:
    average_heatmap_left = np.mean(np.array(all_aligned_gfp_data_left), axis=0)
    average_heatmap_left_normalized = cv2.normalize(average_heatmap_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("左側の有効な馬蹄形が全く検出されませんでした。")
    average_heatmap_left_normalized = np.zeros(target_size, dtype=np.uint8)

average_heatmap_right = None
if all_aligned_gfp_data_right:
    average_heatmap_right = np.mean(np.array(all_aligned_gfp_data_right), axis=0)
    average_heatmap_right_normalized = cv2.normalize(average_heatmap_right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("右側の有効な馬蹄形が全く検出されませんでした。")
    average_heatmap_right_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 5. 可視化 ---
# 表示する画像の総数を動的に計算
# オリジナル画像数 + debug画像数 + 左ヒートマップ + 右ヒートマップ
num_plots = len(processed_images_for_plot) + len(debug_images) + 2

plt.figure(figsize=(num_plots * 4, 7)) # プロット数に応じて全体のサイズを調整

plot_index = 1

# オリジナル画像のプロット（切り抜き済み）
for i, img_to_plot in enumerate(processed_images_for_plot):
    plt.subplot(1, num_plots, plot_index)
    plt.imshow(img_to_plot)
    plt.title(f'Original Image {i+1} \n(Cropped)')
    plt.axis('off')
    plot_index += 1

# Debug画像のプロット（中心線と分類された輪郭）
for i, debug_img_to_plot in enumerate(debug_images):
    plt.subplot(1, num_plots, plot_index)
    plt.imshow(debug_img_to_plot)
    plt.title(f'Debug Contours {i+1} \n(Red: Left, Blue: Right)')
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