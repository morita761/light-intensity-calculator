import gptFunc
import horseShoeShapeRecog.Dog as horse
# import horseShoeShapeRecog.gptFunc as horse
import cv2
import tempHorseshoe as tempH
import matchesTemplate as match
import brightnessMeasure as bright

if __name__ == "__main__":
    image = cv2.imread("pic/up_Fz_green_stronger.tif")  # 入力画像を読み込み
    # ウィンドウを作成
    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Sample Image2', cv2.WINDOW_NORMAL)

    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = image.shape[0] //2
    width = image.shape[1] //2
    
    # ウィンドウサイズを変更，cv2.destroyAllWindows()をするまで使用できる
    cv2.resizeWindow('Sample Image', width, height)
    cv2.resizeWindow('Sample Image2', width, height)
    
    cv2.imshow('Sample Image', image)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    green_mask = gptFunc.extract_green_object(image)
    cv2.imshow('Sample Image', green_mask)
    # cv2.waitKey(0) 
    

    left_mask, right_mask = gptFunc.split_left_right(green_mask)
    cv2.imshow('Sample Image2', left_mask)
    cv2.waitKey(0) 

    cv2.imshow('Sample Image', right_mask)
    # cv2.waitKey(0)

    horseshoe_left, adapt, close, cin = horse.extract_horseshoe_shape(left_mask)
    # horseshoe_left, adapt, close, cin = horse.extract_horseshoe_shape(right_mask)

    cv2.imshow('Sample Image', horseshoe_left)
    # cv2.imshow('Sample Image2', mask)
    cv2.waitKey(0)

    cv2.imshow('Sample Image', adapt)
    # cv2.imshow('Sample Image2', mask)
    cv2.waitKey(0)

    cv2.imshow('Sample Image', close)
    # cv2.imshow('Sample Image2', mask)
    cv2.waitKey(0)

    # ======================= 馬蹄形を抽出 =======================
    valid_horseshoes = []
    temp = tempH.templateHorseshoe()
    for cnt in cin:
        if match.matches_any_template(cnt=cnt, template_list=temp, threshold=0.15):
            valid_horseshoes.append(cnt)
    
    for cnt in valid_horseshoes:
        cv2.drawContours(image, [cnt], -1, 255)
    cv2.imshow('Sample Image', image)
    # cv2.imshow('Sample Image2', mask)
    cv2.waitKey(0)

    # ======================= 輝度を測定 =======================
    offset_x = 21
    offset_y = 10
    radius = 10
    brightness_data = bright.measure_brightness_regions(image, valid_horseshoes, radius=radius, offset_x=offset_x, offset_y=offset_y)

    for b in brightness_data:
        print(f"中心: {b['center']}, 左下の輝度: {b['left_brightness']:.2f}, 右下の輝度: {b['right_brightness']:.2f}")

    for b in brightness_data:
        cx, cy = b['center']
        cv2.circle(image, (cx - offset_x, cy + offset_y), radius, (255, 0, 0), 2)  # 左：青
        cv2.circle(image, (cx + offset_x, cy + offset_y), radius, (0, 0, 255), 2)  # 右：赤
    cv2.imshow('Sample Image', image)
    # cv2.imshow('Sample Image2', mask)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
