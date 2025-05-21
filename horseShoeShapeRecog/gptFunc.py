import cv2
import numpy as np

# 使用中
# 頻出関数，green->2値化，これは統一でいいはず
def green_mask_func(image):
    # BGR → HSV に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 緑色の範囲（HSV値で調整可能）(opencvの値範囲 緑:30~90, S:64~255, V:0~255)
    # 彩度(S)は下限を下げるほど薄い色検出，明度(V)は下げるほど暗い部分も検出
    # lower_green = np.array([40, 50, 50])
    lower_green = np.array([40, 90, 75])
    upper_green = np.array([80, 255, 255])

    # 緑色のマスクを作成
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

# ステップ 3: 馬蹄形（三日月型）の抽出（仮：最大の輪郭を仮定）
def extract_horseshoe_shape(image):
    # green-> gray scale
    mask = green_mask_func(image=image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horseshoe_mask = np.zeros_like(mask)

    if not contours:
        return horseshoe_mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 100:  # 小さいゴミを除外
        return horseshoe_mask
    
    cv2.drawContours(horseshoe_mask, [largest_contour], -1, 255, -1)
    print(type(horseshoe_mask))
    return horseshoe_mask