import cv2
import numpy as np

def extract_green_object(image, coefficient_top = 0.2,
    coefficient_bottom = 0.8,
    coefficient_left = 0.25,
    coefficient_right = 0.66):
    # 画像のどの割合を削除するか，手動で設定する

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