import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 画像の読み込み
image = cv2.imread('mask.tif')
if image is None:
    print("画像を読み込めませんでした。パスを確認してください。")
    exit()

# 例としてダミー画像を生成 (複数の馬蹄形を持つように変更)
# height, width = 600, 800
# image = np.zeros((height, width, 3), dtype=np.uint8)

# # GFP蛍光を模倣 (中央が明るい、ランダムなノイズも追加)
# for y in range(height):
#     for x in range(width):
#         # より複雑な輝度分布をシミュレート
#         intensity = 128 + 100 * np.sin(x * 0.05) * np.cos(y * 0.03) + np.random.randint(-20, 20)
#         image[y, x, 1] = np.clip(intensity, 0, 255).astype(np.uint8) # 緑チャネルに輝度を設定

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
#         angle = np.pi * 1.5 * i / num_points # 馬蹄形の上半分を表現
#         x = int(center_x + radius_x * np.cos(angle) + np.random.randint(-5, 5))
#         y = int(center_y + radius_y * np.sin(angle) + np.random.randint(-5, 5))
#         points.append((x, y))
#     # 線を描画
#     for i in range(len(points) - 1):
#         cv2.line(image, points[i], points[i+1], (255, 0, 0), 3) # 青色で線を描画
#     # 馬蹄形を閉じるための線
#     cv2.line(image, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 3)


# 2. 馬蹄形の領域の特定とマスクの作成
blue_channel = image[:, :, 0] # 青色チャネルを抽出

# 青い線を二値化
# 閾値は画像によって調整が必要
_, binary_blue = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY)

# 形態学的処理で線の途切れを補完（例: 膨張）
kernel = np.ones((5,5), np.uint8)
dilated_blue = cv2.dilate(binary_blue, kernel, iterations=2)
# 穴埋め処理も追加するとより確実 (輪郭の内側を確実に埋めるため)
dilated_blue = cv2.morphologyEx(dilated_blue, cv2.MORPH_CLOSE, kernel, iterations=2)


# 輪郭検出 (全ての外部輪郭を検出)
contours, _ = cv2.findContours(dilated_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. GFP蛍光チャネルの分離と輝度測定
gfp_channel = image[:, :, 1] # 緑色チャネルを抽出 (GFP蛍光)

# ヒートマップ表示用の全体画像を作成 (初期値は0)
full_heatmap_data = np.zeros(image.shape[:2], dtype=np.float32)

# 各馬蹄形についてループ処理
valid_horseshoe_count = 0
for i, contour in enumerate(contours):
    # ある程度の面積を持つ輪郭のみを処理 (ノイズ除去)
    if cv2.contourArea(contour) < 500: # 面積の閾値は調整が必要
        continue

    valid_horseshoe_count += 1
    
    # 各輪郭に対するマスクを作成
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED) # 輪郭の内側を塗りつぶしてマスク作成

    # マスクを適用して馬蹄形の内側の輝度のみを取得
    gfp_intensity_masked = cv2.bitwise_and(gfp_channel, gfp_channel, mask=mask)

    # ヒートマップ表示用のデータに追加
    full_heatmap_data[mask == 255] = gfp_intensity_masked[mask == 255]

# 4. 輝度分布のヒートマップ化
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image with Multiple Horsehoe Shapes')
plt.axis('off')

plt.subplot(1, 2, 2)
# ヒートマップを元の画像の上に重ねて表示
# ここでは、元の画像は背景として薄く表示し、その上にGFP輝度ヒートマップを重ねます。
plt.imshow(image, cmap='gray', alpha=0.0) # 元の画像を透明にして、輝度分布をより強調
sns.heatmap(full_heatmap_data, cmap='viridis', alpha=0.7, cbar=True, ax=plt.gca(),
            xticklabels=False, yticklabels=False)
plt.title(f'GFP Intensity Heatmap within {valid_horseshoe_count} Horseshoes')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"検出された有効な馬蹄形の数: {valid_horseshoe_count}")