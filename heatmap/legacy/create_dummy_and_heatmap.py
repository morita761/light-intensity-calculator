import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import cv2
import numpy as np

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
    # マスク外は黒になる
    clipped_image = cv2.bitwise_and(image, image, mask=clip_mask)
    
    # もし、余白をなくして純粋に切り抜かれた矩形領域が欲しい場合は以下のようにする
    # clipped_image = image[y_start:y_end, x_start:x_end].copy()

    return clipped_image


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
    # cv2.findNonZero はマスク内の非ゼロピクセルの座標を返す
    # それらが一つもなければNoneを返すため、条件分岐で対応
    if cv2.countNonZero(green_mask) > 0:
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(green_mask))
    else:
        # 緑色領域が見つからない場合は、画像全体の中心で分割するなどの代替処理
        print("警告: 画像内に緑色領域が見つかりませんでした。画像全体の中心で分割します。")
        x, y, w, h = 0, 0, width, height # 画像全体を対象とする

    # 緑色領域の中心X座標を計算
    green_center_x = x + w // 2

    # 画像を左右に分割
    # これらの画像は元の画像と同じサイズですが、それぞれの半分のみにデータが含まれ、
    # もう半分は黒になります。
    left_half_image = np.zeros_like(image)
    # green_center_x の左側をコピー
    left_half_image[:, :green_center_x] = image[:, :green_center_x]

    right_half_image = np.zeros_like(image)
    # green_center_x の右側をコピー
    right_half_image[:, green_center_x:] = image[:, green_center_x:]

    return left_half_image, right_half_image

# 1. 画像の読み込み
image = cv2.imread('mask.tif')
if image is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit()

green_mask = extract_region(image, coefficient_left=0.28, coefficient_right=0.6, coefficient_bottom=0.9, coefficient_top=0.2)
cv2.imshow('Sample Image', green_mask)
cv2.waitKey(10*1000)

# right, image = split_left_right(green_mask)
image,left = split_left_right(green_mask)
cv2.imshow('Sample Image', image)
cv2.waitKey(10*1000)

# image = split_left_right(image)

# --- 1. 画像の読み込み（ダミー画像生成） ---
# height, width = 600, 800
# image = np.zeros((height, width, 3), dtype=np.uint8)

# # GFP蛍光を模倣 (中央が明るい、ランダムなノイズも追加)
# for y in range(height):
#     for x in range(width):
#         intensity = 128 + 100 * np.sin(x * 0.05) * np.cos(y * 0.03) + np.random.randint(-20, 20)
#         image[y, x, 1] = np.clip(intensity, 0, 255).astype(np.uint8)

# # 複数の青い馬蹄形を模倣
# horseshoe_definitions = [
#     {'center_x': 200, 'center_y': 200, 'radius_x': 80, 'radius_y': 60},
#     {'center_x': 500, 'center_y': 150, 'radius_x': 70, 'radius_y': 50},
#     {'center_x': 300, 'center_y': 450, 'radius_x': 90, 'radius_y': 70},
#     {'center_x': 700, 'center_y': 400, 'radius_x': 85, 'radius_y': 65}
# ]

# for h_def in horseshoe_definitions:
#     center_x, center_y = h_def['center_x'], h_def['center_y']
#     radius_x, radius_y = h_def['radius_x'], h_def['radius_y']
#     num_points = 100
#     points = []
#     for i in range(num_points):
#         angle = np.pi * 1.5 * i / num_points
#         x = int(center_x + radius_x * np.cos(angle) + np.random.randint(-5, 5))
#         y = int(center_y + radius_y * np.sin(angle) + np.random.randint(-5, 5))
#         points.append((x, y))
#     for i in range(len(points) - 1):
#         cv2.line(image, points[i], points[i+1], (255, 0, 0), 3)
#     cv2.line(image, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 3)

# --- 2. 各馬蹄形の領域の特定とマスクの作成 ---
blue_channel = image[:, :, 0]
_, binary_blue = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
dilated_blue = cv2.dilate(binary_blue, kernel, iterations=2)
dilated_blue = cv2.morphologyEx(dilated_blue, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(dilated_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gfp_channel = image[:, :, 1] # 緑色チャネルを抽出

# --- 3. 各馬蹄形内のGFP輝度データの抽出と正規化 ---
# すべての馬蹄形をこのサイズにリサイズして平均化します
target_size = (100, 100) # 例: 100x100ピクセルに統一
aligned_gfp_data = []

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
    # 輪郭をクロップした画像内の相対座標に変換して描画
    rel_contour = contour - (x, y)
    cv2.drawContours(single_horseshoe_mask, [rel_contour], -1, 255, cv2.FILLED)

    # マスクを適用して、馬蹄形内部の輝度のみを取得
    masked_cropped_gfp = cv2.bitwise_and(cropped_gfp, cropped_gfp, mask=single_horseshoe_mask)

    # リサイズ (双三次補間などで滑らかに)
    # ここでは、マスク外（黒い部分）もリサイズに含まれてしまうため、後でマスクを再適用するか、
    # マスクされた領域を直接変形するなどの工夫が必要になる場合があります。
    # 最も単純なアプローチとして、四角い領域をリサイズします。
    resized_gfp = cv2.resize(masked_cropped_gfp, target_size, interpolation=cv2.INTER_LINEAR)
    
    # リサイズ後に輝度値の範囲を元のままにするためにクリップ
    resized_gfp = np.clip(resized_gfp, 0, 255) # 0-255の範囲にクリップ

    # ここで、リサイズされた画像に再度マスクを適用し、馬蹄形外を確実に0にする
    # これには、代表となる馬蹄形マスクをtarget_sizeで作成し、各リサイズ済み輝度データに適用する必要があります。
    # 簡単のため、今回はリサイズされた輝度データそのままを扱うが、より厳密にはマスクの変形・適用が必要。
    
    aligned_gfp_data.append(resized_gfp)
    valid_horseshoe_count += 1

print(f"有効な馬蹄形の数: {valid_horseshoe_count}")

# --- 4. 輝度データの平均化 ---
# if aligned_gfp_data:
#     # リスト内のすべてのNumPy配列をスタックし、平均を計算
#     average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0)
#     # 必要に応じて正規化（0-1または0-255）
#     average_heatmap_normalized = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# else:
#     print("有効な馬蹄形が検出されませんでした。")
#     average_heatmap_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 4. 輝度データの平均化 ---
if aligned_gfp_data:
    average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0)

    # デバッグポイント3: Matplotlibで直接表示 (カラーマップ適用)
    plt.figure(figsize=(6, 6))
    plt.imshow(average_heatmap, cmap='viridis') # 'viridis'などのカラーマップがおすすめ
    plt.colorbar(label='Average GFP Intensity')
    plt.title("Debug: Average Heatmap (float, raw)")
    plt.axis('off')
    plt.show() # ウィンドウが表示され、閉じると次の処理に進む

    average_heatmap_normalized = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # デバッグポイント4: Matplotlibで正規化後を表示
    # この段階のヒートマップでもよいかも
    plt.figure(figsize=(6, 6))
    plt.imshow(average_heatmap_normalized, cmap='viridis', vmin=0, vmax=255)
    plt.colorbar(label='Normalized Average GFP Intensity')
    plt.title("Debug: Average Heatmap (uint8, normalized)")
    plt.axis('off')
    plt.show()

else:
    print("有効な馬蹄形が検出されませんでした。")
    average_heatmap_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 5. 代表馬蹄形へのヒートマップ表示 ---
# 代表となる馬蹄形のマスクを生成 (平均化された輝度を表示するためのベース)
# これは、平均的な馬蹄形の形状を模倣して手動で作成することもできます。
# ここでは、簡略化のため、単純な楕円形（馬蹄形に近い）を例示します。
# 実際に画像ファイルとして保存（デモンストレーションのため）

# 自身で作成した代表馬蹄形画像を読み込む
# 代表馬蹄形画像がRGBの場合、グレースケールに変換してから使用
# 代表馬蹄形画像は、target_sizeと同じサイズであるか、リサイズが必要になります。
dummy_rep_horseshoe_path = 'horsehoe.png'
representative_horseshoe_mask_raw = cv2.imread(dummy_rep_horseshoe_path, cv2.IMREAD_GRAYSCALE)

if representative_horseshoe_mask_raw is None:
    print(f"代表馬蹄形画像を読み込めませんでした: {dummy_rep_horseshoe_path}")
    print("パスとファイル形式を確認してください。")
    exit()

# representative_mask = np.zeros(target_size, dtype=np.uint8)
# # 平均的な馬蹄形を表現する楕円やカスタム形状を描画
# # 例えば、中心から少し外れた下部を切り欠くなど
# cv2.ellipse(representative_mask, (target_size[0]//2, target_size[1]//2 + target_size[1]//10), 
#             (target_size[0]//2 - 10, target_size[1]//2 - 10), 
#             0, 0, 360, 255, -1) # 全体を楕円で埋める
# 馬蹄形の下部を切り欠く（簡略的な表現）
# cv2.rectangle(representative_mask, (target_size[0]//4, target_size[1]//2 + target_size[1]//5), 
            #   (target_size[0]*3//4, target_size[1]), 0, -1)
# target_sizeにリサイズ（必要に応じて）
representative_mask = cv2.resize(representative_horseshoe_mask_raw, target_size, interpolation=cv2.INTER_LINEAR)
# 確実に二値マスクとして扱う（0または255）
_, representative_horseshoe_mask = cv2.threshold(representative_mask, 127, 255, cv2.THRESH_BINARY)

# 平均ヒートマップを代表マスクに適用 (マスク外は0にする)
final_average_heatmap = cv2.bitwise_and(average_heatmap_normalized, average_heatmap_normalized, mask=representative_mask)

# --- 可視化 ---
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image with Multiple Horsehoe Shapes')
plt.axis('off')

plt.subplot(1, 2, 2)
# 代表馬蹄形マスクの輪郭線を表示
representative_mask_contours, _ = cv2.findContours(representative_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
display_mask_outline = np.zeros((*target_size, 3), dtype=np.uint8)
cv2.drawContours(display_mask_outline, representative_mask_contours, -1, (255, 255, 255), 2) # 白い線で輪郭を描画

# ヒートマップをプロット
sns.heatmap(final_average_heatmap, cmap='viridis', alpha=0.9, cbar=True, ax=plt.gca(),
            xticklabels=False, yticklabels=False)
plt.imshow(display_mask_outline, alpha=0.5) # マスクの輪郭を重ねて表示

plt.title('Average GFP Intensity Heatmap on Representative Horseshoe')
plt.axis('off')

plt.tight_layout()
plt.show()