# 馬蹄形の模式図を自分で描く
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 画像の読み込み（ダミー画像生成） ---
height, width = 600, 800
image = np.zeros((height, width, 3), dtype=np.uint8)

# GFP蛍光を模倣 (中央が明るい、ランダムなノイズも追加)
for y in range(height):
    for x in range(width):
        intensity = 128 + 100 * np.sin(x * 0.05) * np.cos(y * 0.03) + np.random.randint(-20, 20)
        image[y, x, 1] = np.clip(intensity, 0, 255).astype(np.uint8)

# 複数の青い馬蹄形を模倣
horseshoe_definitions = [
    {'center_x': 200, 'center_y': 200, 'radius_x': 80, 'radius_y': 60},
    {'center_x': 500, 'center_y': 150, 'radius_x': 70, 'radius_y': 50},
    {'center_x': 300, 'center_y': 450, 'radius_x': 90, 'radius_y': 70},
    {'center_x': 700, 'center_y': 400, 'radius_x': 85, 'radius_y': 65}
]

for h_def in horseshoe_definitions:
    center_x, center_y = h_def['center_x'], h_def['center_y']
    radius_x, radius_y = h_def['radius_x'], h_def['radius_y']
    num_points = 100
    points = []
    for i in range(num_points):
        angle = np.pi * 1.5 * i / num_points
        x = int(center_x + radius_x * np.cos(angle) + np.random.randint(-5, 5))
        y = int(center_y + radius_y * np.sin(angle) + np.random.randint(-5, 5))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i+1], (255, 0, 0), 3)
    cv2.line(image, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 3)

# --- 2. 各馬蹄形の領域の特定とマスクの作成 ---
blue_channel = image[:, :, 0]
_, binary_blue = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
dilated_blue = cv2.dilate(binary_blue, kernel, iterations=2)
dilated_blue = cv2.morphologyEx(dilated_blue, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(dilated_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gfp_channel = image[:, :, 1] # 緑色チャネルを抽出

# --- 3. 各馬蹄形内のGFP輝度データの抽出と正規化 ---
target_size = (100, 100) # 馬蹄形をこのサイズにリサイズして平均化
aligned_gfp_data = []

valid_horseshoe_count = 0
for contour in contours:
    if cv2.contourArea(contour) < 500: # 小さすぎるノイズ輪郭を除外 (閾値は調整)
        continue

    x, y, w, h = cv2.boundingRect(contour)
    cropped_gfp = gfp_channel[y:y+h, x:x+w]

    single_horseshoe_mask = np.zeros((h, w), dtype=np.uint8)
    rel_contour = contour - (x, y)
    cv2.drawContours(single_horseshoe_mask, [rel_contour], -1, 255, cv2.FILLED)

    masked_cropped_gfp = cv2.bitwise_and(cropped_gfp, cropped_gfp, mask=single_horseshoe_mask)
    resized_gfp = cv2.resize(masked_cropped_gfp, target_size, interpolation=cv2.INTER_LINEAR)
    resized_gfp = np.clip(resized_gfp, 0, 255)
    
    aligned_gfp_data.append(resized_gfp)
    valid_horseshoe_count += 1

print(f"検出された有効な馬蹄形の数: {valid_horseshoe_count}")

# --- 4. 輝度データの平均化 ---
if aligned_gfp_data:
    average_heatmap = np.mean(np.array(aligned_gfp_data), axis=0)
    average_heatmap_normalized = cv2.normalize(average_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    print("有効な馬蹄形が検出されませんでした。")
    average_heatmap_normalized = np.zeros(target_size, dtype=np.uint8)

# --- 5. 自身で作成した代表馬蹄形画像の読み込みとマスクとしての利用 ---
# ここに、あなたが作成した馬蹄形の形状画像パスを指定してください
# 例: 'my_horseshoe_shape.png' または 'my_horseshoe_shape.tif'
# この画像は、黒背景に白い馬蹄形の形状が描かれている白黒（またはグレースケール）画像であると仮定します。

# ダミーの代表馬蹄形画像を作成 (実際にファイルを用意する代わりにコードで生成)
# ユーザーはこれを実際のファイルパスに置き換える
dummy_rep_horseshoe_path = 'dummy_representative_horseshoe.png'
dummy_rep_horseshoe = np.zeros(target_size, dtype=np.uint8)
# より複雑な形状を模倣するために複数の曲線を描画
cv2.ellipse(dummy_rep_horseshoe, (target_size[0]//2, target_size[1]//2), 
            (target_size[0]//2 - 5, target_size[1]//2 - 5), 
            0, 0, 360, 255, -1) # 外側
cv2.ellipse(dummy_rep_horseshoe, (target_size[0]//2, target_size[1]//2 - 10), 
            (target_size[0]//2 - 25, target_size[1]//2 - 25), 
            0, 0, 360, 0, -1) # 内側をくり抜く（馬蹄形の上部）

# 馬蹄形の下部を切り欠くための矩形
cv2.rectangle(dummy_rep_horseshoe, (target_size[0]//4, target_size[1]*3//4), 
              (target_size[0]*3//4, target_size[1]), 0, -1)
# 実際に画像ファイルとして保存（デモンストレーションのため）
cv2.imwrite(dummy_rep_horseshoe_path, dummy_rep_horseshoe)

# 自身で作成した代表馬蹄形画像を読み込む
# 代表馬蹄形画像がRGBの場合、グレースケールに変換してから使用
# 代表馬蹄形画像は、target_sizeと同じサイズであるか、リサイズが必要になります。
representative_horseshoe_mask_raw = cv2.imread(dummy_rep_horseshoe_path, cv2.IMREAD_GRAYSCALE)

if representative_horseshoe_mask_raw is None:
    print(f"代表馬蹄形画像を読み込めませんでした: {dummy_rep_horseshoe_path}")
    print("パスとファイル形式を確認してください。")
    exit()

# target_sizeにリサイズ（必要に応じて）
representative_horseshoe_mask = cv2.resize(representative_horseshoe_mask_raw, target_size, interpolation=cv2.INTER_LINEAR)

# 確実に二値マスクとして扱う（0または255）
_, representative_horseshoe_mask = cv2.threshold(representative_horseshoe_mask, 127, 255, cv2.THRESH_BINARY)


# 平均ヒートマップを代表マスクに適用 (マスク外は0にする)
final_average_heatmap = cv2.bitwise_and(average_heatmap_normalized, average_heatmap_normalized, mask=representative_horseshoe_mask)

# --- 可視化 ---
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image with Multiple Horsehoe Shapes')
plt.axis('off')

plt.subplot(1, 2, 2)
# 代表馬蹄形マスクの輪郭線を表示
# 代表馬蹄形マスクの輪郭を抽出し、表示用の画像に描画
rep_mask_contours, _ = cv2.findContours(representative_horseshoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
display_mask_outline = np.zeros((*target_size, 3), dtype=np.uint8)
cv2.drawContours(display_mask_outline, rep_mask_contours, -1, (255, 255, 255), 2) # 白い線で輪郭を描画

# ヒートマップをプロット
sns.heatmap(final_average_heatmap, cmap='viridis', alpha=0.9, cbar=True, ax=plt.gca(),
            xticklabels=False, yticklabels=False, vmin=0, vmax=255) # 輝度範囲を固定
plt.imshow(display_mask_outline, alpha=0.5) # マスクの輪郭を重ねて表示

plt.title('Average GFP Intensity Heatmap on Custom Representative Horseshoe')
plt.axis('off')

plt.tight_layout()
plt.show()