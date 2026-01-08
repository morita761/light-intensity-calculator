import cv2
import recogBlue
import numpy as np
import unet
import predict
import opencv

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img 

if __name__ == "__main__":
    # image = cv2.imread("../pic/up_Fz_green_stronger.tif")  # 入力画像を読み込み
    image = cv2.imread("../pic/up_Fz_green_stronger_selected_2_blue.tif")  # 入力画像を読み込み
    # pic/up_Fz_green_stronger_selected_2_blue.tif
    # ウィンドウを作成
    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)

    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = image.shape[0] //2
    print(height)
    width = image.shape[1] //2
    print(width)
    # ウィンドウサイズを変更，cv2.destroyAllWindows()をするまで使用できる
    cv2.resizeWindow('Sample Image', width, height)
    
    cv2.imshow('Sample Image', image)
    cv2.waitKey(0) 

    img_tensor, mask_tensor = recogBlue.load_image_and_mask("../pic/up_Fz_green_stronger.tif","../pic/up_Fz_green_stronger_selected_2_blue.tif", (width,height))

    mask_np = mask_tensor.squeeze(0).numpy().astype(np.uint8) * 255  # 0 or 255 に変換
    cv2.imshow('Sample Image', mask_np)
    cv2.waitKey(0) 
    # opencv.show_mask_cv(mask_tensor)
    # unet.model

    predict.predict_single_image(unet.model, "../pic/up_Fz_green_stronger.tif")

    cv2.waitKey(0) 
    cv2.destroyAllWindows()
