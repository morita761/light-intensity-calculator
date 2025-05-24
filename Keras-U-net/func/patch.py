import numpy as np
import cv2

def extract_patches(img, mask, patch_size=256, stride=128):
    patches_img, patches_mask = [], []
    h, w = img.shape[:2]

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = img[y:y+patch_size, x:x+patch_size]
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            # マスクが空でない部分のみ使う，対象が5%以上含まれている（面積でフィルターかけてるから大丈夫なはず）
            # if np.any(patch_mask) / (patch_size * patch_size * 255) > 0.05:
            if np.any(patch_mask):    
                patches_img.append(patch_img)
                patches_mask.append(patch_mask[..., None] / 255.0)  # 正規化
    print(type(np.array(patches_img)))
    print(type(np.array(patches_mask)))
    # cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    # # 読み込んだ画像の高さと幅を取得，整数で割り算
    # height = img.shape[0] //2
    # width = img.shape[1] //2
    # cv2.resizeWindow('Sample Image', width, height)
    # cv2.imshow("Sample Image", np.array(patches_img))
    # cv2.waitKey(3*1000)
    # cv2.imshow("Sample Image", np.array(patches_mask))
    # cv2.waitKey(3*1000)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.array(patches_img), np.array(patches_mask)
