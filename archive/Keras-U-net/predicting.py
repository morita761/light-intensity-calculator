import cv2
from sklearn.model_selection import train_test_split
import func.bluemask as blue
import func.patch as patch
import kerasUnet as K
import func.predict as predict
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import glob
from keras.models import load_model


# --- ファイルパスの取得 ---
# image_files = sorted(glob.glob("./simpledataset/images/*"))
# mask_files = sorted(glob.glob("./simpledataset/masks/*"))
image_files = sorted(glob.glob("../data/img/*.tif"))
mask_files = sorted(glob.glob("../data/masks/*_mask.tif"))
predict_image = image_files[1]

assert len(image_files) == len(mask_files), "画像とマスクの数が一致しません"

img = cv2.imread(image_files[0])
height, width = img.shape[:2]

# --- Dice Loss ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# --- BCE + Dice Loss ---
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

# load_model時に明示的に渡す
model = load_model("unet_model.h5", custom_objects={'combined_loss': combined_loss, 'dice_loss': dice_loss})
# model = load_model("./test-ver-simple/simple_unet_model.h5", custom_objects={'combined_loss': combined_loss, 'dice_loss': dice_loss})

# --- 推論とマスク保存 ---
result_mask = predict.predict_on_test_image(predict_image, model, stride=64, ave_map_threshold=0.3)
padded_mask = np.zeros((height, width), dtype=result_mask.dtype)
h, w = result_mask.shape
padded_mask[:min(height, h), :min(width, w)] = result_mask[:min(height, h), :min(width, w)]

cv2.imwrite("./output/predicted_mask.png", padded_mask)

# --- オーバーレイ出力 ---
def overlay_mask(image_path, result_mask):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    # resizeではなく、中央に貼り付け（padding）
    padded_mask = np.zeros((height, width), dtype=result_mask.dtype)
    h, w = result_mask.shape
    padded_mask[:min(height, h), :min(width, w)] = result_mask[:min(height, h), :min(width, w)]
    overlay = img.copy()
    # mask = cv2.resize(mask, (width, height))
    # mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_color = cv2.applyColorMap(padded_mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
    return blended

cv2.imwrite("./output/overlay_debug.png", overlay_mask(predict_image, result_mask))
