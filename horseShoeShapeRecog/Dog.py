import cv2
import numpy as np
import horseShoeShapeRecog.ellipseAppro as ellip

## 2025/05/22 1:44 使用
def extract_horseshoe_shape(image):
    # Y = 0.299 * R + 0.587 * G + 0.114 * BでRGBから輝度信号Yを算出
    # 人間の視覚特性を考慮したもので、最も明るく感じるG（緑）の係数が大きくなっている。
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("gray: "+ str(type(gray)))

    # またはCLAHE（局所適応ヒストグラム平坦化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 明暗差の影響を軽減し、局所的な境界を強調
    adaptive = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=13, C=1)

    # モルフォロジー処理（境界である白を膨張→収縮で閉じた輪郭に）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)


    # # 輪郭抽出
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(type(contours))
    for cnt in contours:
        # area = cv2.contourArea(cnt)
        # if 250 < area < 900:
        #     # 馬蹄形の大きさでフィルタ
            cv2.drawContours(equalized, [cnt], -1, 255, )
        # if ellip.is_horseshoe_like(cnt):
        #     cv2.drawContours(equalized, [cnt], -1, 255, 4)

    # Canny法
    edges = cv2.Canny(equalized, threshold1=40, threshold2=75)

    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # horseshoe_mask = np.zeros_like(gray)

    return equalized,adaptive, closed, contours