import cv2
from sklearn.model_selection import train_test_split
import func.bluemask as blue
import func.patch as patch
import kerasUnet as K
import func.predict as predict
import sys

# 入力画像とマスク（青線抽出後）を読み込む
# img = cv2.imread("./dataset/images/up_Fz_green_stronger.tif")
# mask = blue.extract_blue_mask("./dataset/masks/up_Fz_green_stronger_mask.tif")

img = cv2.imread("./simpledataset/images/02.png")
mask = blue.extract_blue_mask("./simpledataset/masks/02_mask.png")
height = img.shape[0]
width = img.shape[1]

print("test")
# sys.exit()
X, y = patch.extract_patches(img, mask)
print(X.shape)  # (N, 256, 256, 3)
print(y.shape) # (N, 256, 256, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = K.build_unet()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=8)

# result_mask = predict.predict_on_test_image("./dataset/images/up_Fz_green_stronger.tif", model)
result_mask = predict.predict_on_test_image("./simpledataset/images/02.png", model)
result_mask = cv2.resize(result_mask, (width, height))
cv2.imwrite("predicted_mask.png", result_mask)

print("test")

def overlay_mask(image_path, mask):
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]
    overlay = img.copy()
    mask = cv2.resize(mask, (width, height))
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_color[:height, :width]
    blended = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
    return blended

cv2.imwrite("overlay_debug.png", overlay_mask("./simpledataset/images/02.png", result_mask))
