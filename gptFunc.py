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

# ステップ 1: 緑色オブジェクトの抽出
def extract_green_object(image):
    # 画像のどの割合を削除するか，手動で設定する
    coefficient_top = 0.2
    coefficient_bottom = 0.8
    coefficient_left = 0.25
    coefficient_right = 0.66

    # BGR → HSV に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 緑色の範囲（HSV値で調整可能）
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # 緑色のマスクを作成
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 指定領域（例：上から1/6、左から1/8）を無効化
    height, width = image.shape[:2]
    exclude_mask = np.ones_like(green_mask, dtype=np.uint8) * 255
    exclude_top = int(height*coefficient_top)
    exclude_bottom = int(height*coefficient_bottom)
    exclude_left = int(width*coefficient_left)
    exclude_right = int(width*coefficient_right)

    exclude_mask[:exclude_top, :] = 0  # 上から1/6行を除外
    exclude_mask[exclude_bottom:, :] = 0  # 下から1/6行を除外
    exclude_mask[:, :exclude_left] = 0  # 左から1/8列を除外
    exclude_mask[:, exclude_right:] = 0  # 左から1/8列を除外

    # 排除領域を反映したマスク
    final_mask = cv2.bitwise_and(green_mask, exclude_mask)

    # 緑色だけを抽出
    result = cv2.bitwise_and(image, image, mask=final_mask)
    # print(type(result))

    return result

# ステップ 2: 緑色オブジェクトの左右分割
def split_left_right(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 緑色の範囲（HSV値で調整可能）
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # 緑色のマスクを作成
    mask = cv2.inRange(hsv, lower_green, upper_green)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    left_mask = np.zeros_like(mask)
    right_mask = np.zeros_like(mask)
    left_mask[y:y+h, x:x + w//2] = mask[y:y+h, x:x + w//2]
    right_mask[y:y+h, x + w//2:x + w] = mask[y:y+h, x + w//2:x + w]
    
    # 排除領域を反映したマスク
    left_mask = cv2.bitwise_and(image, image, mask=left_mask)
    right_mask = cv2.bitwise_and(image, image, mask=right_mask)

    # print(type(left_mask))
    return left_mask, right_mask



# ステップ 4: 左右端の輝度を測定　つかってない
def measure_brightness(image, left_mask, right_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left_brightness = np.mean(gray[left_mask > 0]) if np.any(left_mask) else 0
    right_brightness = np.mean(gray[right_mask > 0]) if np.any(right_mask) else 0
    return left_brightness, right_brightness
