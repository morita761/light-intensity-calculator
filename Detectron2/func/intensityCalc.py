import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math

def extract_green_points(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([40, 50, 50])
    lower_green = np.array([40, 90, 75])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    points = cv2.findNonZero(mask)
    return points.reshape(-1, 2), mask

def perform_pca(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_
    direction = pca.components_[0]
    # PCA の方向ベクトルを取得したあとに、Y成分が負なら反転
    if direction[1] < 0:
        direction = -direction

    angle = math.atan2(direction[1], direction[0])  # ラジアン
    angle_deg = np.degrees(angle)
    print(angle_deg)

    # 馬蹄形の方向が -60°～+60° 以内でなければスキップや修正
    # if not (-60 <= angle_deg <= 60):
    #     print(f"角度が {angle_deg:.2f}° で範囲外のため補正またはスキップ")
    #     # 方向を修正したい場合、最も近い境界角度にスナップ（例：±60°）
    #     angle = np.clip(angle, np.radians(-60), np.radians(60))
    #     direction = np.array([np.cos(angle), np.sin(angle)])

    return center, direction

def perform_kmeans(points, k=2):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    return kmeans.labels_, kmeans.cluster_centers_


def draw_results(image, center, direction, cluster_centers, radius=5):
    result = image.copy()
    green_channel = image[:, :, 1]  # Gチャンネル
    # height, width = green_channel.shape

    # 向き補正：Y方向が下（正）になるようにベクトルを反転
    if direction[1] < 0:
        print("inverse")
        direction = -direction

    # 角度が -60〜+60度の範囲内にあるか確認
    angle = np.arctan2(direction[1], direction[0])
    angle_deg = np.degrees(angle)
    if not (-60 <= angle_deg <= 60):
        angle = np.clip(angle, np.radians(-60), np.radians(60))
        direction = np.array([np.cos(angle), np.sin(angle)])

    # PCA主軸に投影して左右クラスタに分割
    # クラスタリングの補強としても利用できる
    points = np.vstack(cluster_centers)
    projected = (points - center) @ direction
    left_cluster = points[projected < 0]
    right_cluster = points[projected >= 0]

    # 主軸描画
    length = 100
    pt1 = tuple(np.round(center).astype(int))
    pt2 = tuple(np.round(center + direction * length).astype(int))
    cv2.arrowedLine(result, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)

    # クラスタ中心描画
    for c in left_cluster:
        cv2.circle(result, tuple(np.round(c).astype(int)), 8, (0, 255, 0))
    for c in right_cluster:
        cv2.circle(result, tuple(np.round(c).astype(int)), 8, (255, 0, 0))

    left_vals = []
    for c in left_cluster:
        x, y = np.round(c).astype(int)
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        region = green_channel[mask == 255]
        if region.size > 0:
            left_vals.append(np.mean(region))
        cv2.circle(result, (x, y), radius, (0, 255, 0), 1)

    right_vals = []
    for c in right_cluster:
        x, y = np.round(c).astype(int)
        mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        region = green_channel[mask == 255]
        if region.size > 0:
            right_vals.append(np.mean(region))
        cv2.circle(result, (x, y), radius, (255, 0, 0), 1)

    # 平均値として返す（空なら0.0）
    left_mean = float(np.mean(left_vals)) if left_vals else 0.0
    right_mean = float(np.mean(right_vals)) if right_vals else 0.0

    return result, left_cluster, right_cluster, left_mean, right_mean


if __name__ == "__main__":
    # 実行例
    image = cv2.imread("../../data/test_img/006.tif")
    points, mask = extract_green_points(image)
    cv2.imshow('Sample Image', mask)
    cv2.waitKey(0) 
    center, direction = perform_pca(points)
    labels, cluster_centers = perform_kmeans(points)
    result_img, left_cluster, right_cluster, left_brightness, right_brightness = draw_results(image, center, direction, cluster_centers)
    # cv2.imwrite("pca_kmeans_result.png", result_img)
    cv2.imshow('Sample Image', result_img)
    cv2.waitKey(0) 

    print("left bright:", left_brightness[0])
    print("right bright:", right_brightness[0])