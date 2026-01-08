import cv2
import numpy as np
import func.horsesheLike as horse

def extract_blue_mask(label_image_path):
    img = cv2.imread(label_image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel, iterations=4)

    # 青色のHSV範囲（必要に応じて調整）
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 輪郭で囲まれた中を塗りつぶす（青線内をマスクに）
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    filled_mask_x = np.zeros_like(mask)
    # cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    valid_contours = []
    for cnt in contours:
        cv2.drawContours(filled_mask_x, [cnt], -1, 255, thickness=cv2.FILLED) 
        # 機能テスト       
        if horse.is_horseshoe_like(cnt):
            valid_contours.append(cnt)
            # print("true")
        # else:
        #     print("False!!")

    for cnt in valid_contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = img.shape[0] //2
    width = img.shape[1] //2
    cv2.resizeWindow('Sample Image', width, height)
    cv2.imshow("Sample Image", filled_mask)
    cv2.waitKey(3*1000)
    cv2.imshow("Sample Image", filled_mask_x)
    cv2.waitKey(3*1000)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()

    return filled_mask

def extract_blue_neg_mask(label_image_path):
    img = cv2.imread(label_image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # # モルフォロジー処理で線を膨張
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel, iterations=4)

    # 青色のHSV範囲（必要に応じて調整）
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 輪郭で囲まれた中を塗りつぶす（青線内のみ）
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)

    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = img.shape[0] //2
    width = img.shape[1] //2
    cv2.resizeWindow('Sample Image', width, height)
    cv2.imshow("Sample Image", filled_mask)
    cv2.waitKey(3*1000)

    cv2.destroyAllWindows()

    return filled_mask