import cv2
import horseShoeShapeRecog.ellipseAppro as ellip
import horseShoeShapeRecog.Dog as horse

## 2025/05/22 1:44 使用
def templateHorseshoe():
    image = cv2.imread("pic/horseshoe-whiteback.png")  # 入力画像を読み込み
    # ウィンドウを作成
    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
        # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = image.shape[0] //2
    width = image.shape[1] //2
    
    # ウィンドウサイズを変更，cv2.destroyAllWindows()をするまで使用できる
    # cv2.resizeWindow('Sample Image', width, height)

    cv2.imshow('Sample Image', image)
    cv2.waitKey(0) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # 輪郭抽出
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     # area = cv2.contourArea(cnt)
    #     # if 250 < area < 900:
    #     #     # 馬蹄形の大きさでフィルタ
    #         cv2.drawContours(gray, [cnt], -1, 255, -1)
    return contours
    # cv2.imshow('Sample Image', gray)
    # cv2.waitKey(0)
    
    # # cv2.waitKey(0) 
    # cv2.destroyAllWindows()