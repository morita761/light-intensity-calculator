import cv2
import numpy as np

def show_mask_cv(mask_tensor):
    mask_np = mask_tensor.squeeze(0).numpy().astype(np.uint8) * 255  # 0 or 255 に変換
    cv2.imshow("Mask (Blue Area)", mask_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
