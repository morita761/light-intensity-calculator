import sys

import cv2
import horseShoeShapeRecog.ellipseAppro as ellip
import horseShoeShapeRecog.Dog as horse
import numpy as np

# 自分が作成した馬蹄形をテストして出力するためのコード

IMAGE_PATH = "pic/horseshoe-whiteback.png"

if __name__ == "__main__":
    image = cv2.imread(IMAGE_PATH)  # 入力画像を読み込み
    if image is None:
        print(
            f"エラー: 入力画像を読み込めませんでした: {IMAGE_PATH}\n"
            f"  原因: ファイルが存在しないか、画像として読み込めない形式です。\n"
            f"  対処: pic/ ディレクトリに '{IMAGE_PATH.split('/')[-1]}' を配置するか、"
            f"horseshoe.py 冒頭の IMAGE_PATH を実際の画像パスに変更してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    # ウィンドウを作成
    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
        # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = image.shape[0] //2
    width = image.shape[1] //2
    
    # ウィンドウサイズを変更，cv2.destroyAllWindows()をするまで使用できる
    cv2.resizeWindow('Sample Image', width, height)

    cv2.imshow('Sample Image', image)
    cv2.waitKey(0) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
        # # 輪郭抽出
    valid_contours = []

    white_bg = np.full_like(gray, 255)  # 全ピクセル255（白）の同サイズ画像

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(white_bg, [cnt], -1, 0, -1) 
        # 機能テスト       
        if ellip.is_horseshoe_like(cnt):
            valid_contours.append(cnt)
        else:
            print("False!!")

    cv2.imshow('Sample Image', white_bg)
    cv2.waitKey(0)

    for cnt in valid_contours:
        cv2.drawContours(gray, [cnt], -1, 255, -1)
    
    cv2.imshow('Sample Image', gray)
    cv2.waitKey(0)
    
    # cv2.waitKey(0) 
    cv2.destroyAllWindows()