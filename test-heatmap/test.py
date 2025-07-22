import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# def split_left_right(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # 緑色の範囲（HSV値で調整可能）
#     lower_green = np.array([40, 50, 50])
#     upper_green = np.array([80, 255, 255])

#     # 緑色のマスクを作成
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
#     left_mask = np.zeros_like(mask)
#     right_mask = np.zeros_like(mask)
#     left_mask[y:y+h, x:x + w//2] = mask[y:y+h, x:x + w//2]
#     right_mask[y:y+h, x + w//2:x + w] = mask[y:y+h, x + w//2:x + w]
    
#     # 排除領域を反映したマスク
#     left_mask = cv2.bitwise_and(image, image, mask=left_mask)
#     right_mask = cv2.bitwise_and(image, image, mask=right_mask)

#     # print(type(left_mask))
#     return left_mask
#     # return left_mask, right_mask

# 1. 画像の読み込み
image = cv2.imread('mask.tif')
if image is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit()

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
if aligned_gfp_data:
    # リスト内のすべてのNumPy配列をスタックし、平均を計算
    average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0)
    # 必要に応じて正規化（0-1または0-255）
    average_heatmap_normalized = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("有効な馬蹄形が検出されませんでした。")
    average_heatmap_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 5. 代表馬蹄形へのヒートマップ表示 ---
# 代表となる馬蹄形のマスクを生成 (平均化された輝度を表示するためのベース)
# これは、平均的な馬蹄形の形状を模倣して手動で作成することもできます。
# ここでは、簡略化のため、単純な楕円形（馬蹄形に近い）を例示します。
representative_mask = np.zeros(target_size, dtype=np.uint8)
# 平均的な馬蹄形を表現する楕円やカスタム形状を描画
# 例えば、中心から少し外れた下部を切り欠くなど
cv2.ellipse(representative_mask, (target_size[0]//2, target_size[1]//2 + target_size[1]//10), 
            (target_size[0]//2 - 10, target_size[1]//2 - 10), 
            0, 0, 360, 255, -1) # 全体を楕円で埋める
# 馬蹄形の下部を切り欠く（簡略的な表現）
cv2.rectangle(representative_mask, (target_size[0]//4, target_size[1]//2 + target_size[1]//5), 
              (target_size[0]*3//4, target_size[1]), 0, -1)

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