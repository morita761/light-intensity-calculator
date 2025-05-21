import cv2
import numpy as np
## 2025/05/22 1:44 使用
def measure_brightness_regions(image, contours, radius=15, offset_x=20, offset_y=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_results = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        # 重心座標
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # オフセットを加えて右下・左下に円を作成　あ～これか
        # しっかり馬蹄形つかめてないときついのか，左右の大きさ取って計算するか，重心はあるのか
        # offset_x = 20  # 横方向のずらし量（必要に応じて調整）
        # offset_y = 20  # 縦方向のずらし量（必要に応じて調整）
        offset_x = offset_x
        offset_y = offset_y

        left_center  = (cx - offset_x, cy + offset_y)
        right_center = (cx + offset_x, cy + offset_y)

        # マスク作成（1枚ごとに）
        left_mask  = np.zeros_like(gray, dtype=np.uint8)
        right_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(left_mask,  left_center, radius, 255, -1)
        cv2.circle(right_mask, right_center, radius, 255, -1)

        # 各マスク内の平均輝度を計算
        left_mean  = cv2.mean(gray, mask=left_mask)[0]
        right_mean = cv2.mean(gray, mask=right_mask)[0]

        # 結果を記録（またはprintしてもよい）
        brightness_results.append({
            'center': (cx, cy),
            'left_brightness': left_mean,
            'right_brightness': right_mean,
        })

    return brightness_results
