import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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
    入力画像を、画像内の緑色領域の幅の中央で左右に二分割します。

    Args:
        image (np.array): 左右に分割する画像（BGR形式）。
                          これは `extract_region` の戻り値であると想定され、
                          画像内に緑色のオブジェクトが含まれている必要があります。

    Returns:
        tuple: (left_half_image, right_half_image)
               left_half_image (np.array): 緑色領域の左半分。
               right_half_image (np.array): 緑色領域の右半分。
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

    # 画像を左右に分割
    left_half_image = np.zeros_like(image)
    left_half_image[:, :green_center_x] = image[:, :green_center_x]

    right_half_image = np.zeros_like(image)
    right_half_image[:, green_center_x:] = image[:, green_center_x:]

    return left_half_image, right_half_image

# --- メイン処理 ---
# 1. 複数の画像のファイル名を指定
image_files = ['002_stronger_mask.tif', '003_stronger_mask.tif'] # ここに処理したい画像ファイル名を追加

all_aligned_gfp_data = [] # すべての画像から得られるGFPデータを格納するリスト
processed_images_for_plot = [] # プロット用の切り抜かれたオリジナル画像を格納するリスト

target_size = (100, 100) # 例: 100x100ピクセルに統一

for file_name in image_files:
    # 1. 画像の読み込み
    image_original = cv2.imread(file_name)
    if image_original is None:
        print(f"画像を読み込めませんでした: {file_name}。パスを確認してください。")
        continue

    green_mask_image = extract_region(image_original, coefficient_left=0.28, coefficient_right=0.75, coefficient_bottom=0.92, coefficient_top=0.09)

    right_image, left_image = split_left_right(green_mask_image)
    
    # 今回はright_imageのみ処理する場合
    image = right_image
    # image = left_image

    # debug用
    # cv2.imshow('Sample Image', image)
    # cv2.waitKey(10*1000)
    # sys.exit()

    # 黒い領域を切り抜いてプロット用に保存
    # cropped_image_for_plot = extract_region(image, coefficient_left=0.28, coefficient_right=0.75, coefficient_bottom=0.8, coefficient_top=0.1)
    processed_images_for_plot.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Matplotlib用にBGRからRGBに変換

    # --- 2. 各馬蹄形の領域の特定とマスクの作成 ---
    blue_channel = image[:, :, 0]
    _, binary_blue = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    dilated_blue = cv2.dilate(binary_blue, kernel, iterations=2)
    dilated_blue = cv2.morphologyEx(dilated_blue, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gfp_channel = image[:, :, 1] # 緑色チャネルを抽出

    # --- 3. 各馬蹄形内のGFP輝度データの抽出と正規化 ---
    valid_horseshoe_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 100: # 小さすぎるノイズ輪郭を除外
            continue

        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)

        # バウンディングボックス内の輝度データを抽出
        cropped_gfp = gfp_channel[y:y+h, x:x+w]

        # この馬蹄形のみのマスクを作成
        single_horseshoe_mask = np.zeros((h, w), dtype=np.uint8)
        rel_contour = contour - (x, y)
        cv2.drawContours(single_horseshoe_mask, [rel_contour], -1, 255, cv2.FILLED)

        # マスクを適用して、馬蹄形内部の輝度のみを取得
        masked_cropped_gfp = cv2.bitwise_and(cropped_gfp, cropped_gfp, mask=single_horseshoe_mask)

        # リサイズ (双三次補間などで滑らかに)
        resized_gfp = cv2.resize(masked_cropped_gfp, target_size, interpolation=cv2.INTER_LINEAR)
        resized_gfp = np.clip(resized_gfp, 0, 255) # 0-255の範囲にクリップ

        all_aligned_gfp_data.append(resized_gfp)
        valid_horseshoe_count += 1
    
    print(f"{file_name} から有効な馬蹄形が {valid_horseshoe_count} 個検出されました。")


# --- 4. すべての輝度データの平均化 ---
if all_aligned_gfp_data:
    average_heatmap = np.mean(np.array(all_aligned_gfp_data), axis=0)

    average_heatmap_normalized = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("有効な馬蹄形が全く検出されませんでした。")
    average_heatmap_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 5. 可視化 ---
plt.figure(figsize=(15, 7))

# オリジナル画像のプロット（切り抜き済み）
for i, img_to_plot in enumerate(processed_images_for_plot):
    plt.subplot(1, 2 + len(image_files) - 1, i + 1) # ヒートマップの分も考慮してサブプロット数を調整
    plt.imshow(img_to_plot)
    plt.title(f'Original Image {i+1} (Cropped)')
    plt.axis('off')

# ヒートマップのプロット
plt.subplot(1, 2 + len(image_files) - 1, len(image_files) + 1)
plt.imshow(average_heatmap_normalized, cmap='viridis', vmin=0, vmax=255)
plt.colorbar(label='Normalized Average GFP Intensity')
plt.title('Average GFP Intensity Heatmap on Representative Horseshoe (Combined)')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()