import cv2
from sklearn.model_selection import train_test_split
import func.bluemask as blue
import func.patch as patch
import kerasUnet as K
import func.predict as predict
import sys
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from pathlib import Path

# --- Dice Loss 定義 ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# --- BCE + Dice Loss 合成関数 ---
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

# --- 画像とマスク読み込み ---
# img_file = Path("./simpledataset/images/02.png")
# mask_file = Path("./simpledataset/masks/02_mask.png")
img_file = Path("./dataset/images/up_Fz_green_stronger.tif")
mask_file = Path("./dataset/masks/up_Fz_green_stronger_mask.tif")
img = cv2.imread(img_file)
mask = blue.extract_blue_mask(mask_file)
height, width = img.shape[:2]

# --- パッチ抽出 ---
X, y = patch.extract_patches(img, mask)
print(X.shape)
print(y.shape)

# --- 訓練/検証分割 ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --- モデル構築とコンパイル ---
model = K.build_unet()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

# --- EarlyStopping コールバック ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- 学習エポックの推移 コールバック ---
class PrintPredMeanCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(X_val[:1], verbose=0)
        print(f"Epoch {epoch}: pred.mean() = {np.mean(pred)}")

# --- 学習（エポック数増加） ---
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8,
          callbacks=[early_stop, PrintPredMeanCallback()])

# --- 推論とマスク保存 ---
result_mask = predict.predict_on_test_image(img_file, model)
padded_mask = np.zeros((height, width), dtype=result_mask.dtype)
h, w = result_mask.shape
padded_mask[:min(height, h), :min(width, w)] = result_mask[:min(height, h), :min(width, w)]
# result_mask = cv2.resize(result_mask, (width, height))
cv2.imwrite("predicted_mask.png", padded_mask)

# --- オーバーレイ出力 ---
def overlay_mask(image_path, result_mask):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    # マスクをcropしてresizeする
    # cropped = result_mask[:624, :1254]  # オーバー分を切り捨て
    # あるいはresizeではなく、中央に貼り付け（padding）
    padded_mask = np.zeros((height, width), dtype=result_mask.dtype)
    h, w = result_mask.shape
    padded_mask[:min(height, h), :min(width, w)] = result_mask[:min(height, h), :min(width, w)]
    overlay = img.copy()
    # mask = cv2.resize(mask, (width, height))
    # mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_color = cv2.applyColorMap(padded_mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
    return blended

cv2.imwrite("overlay_debug.png", overlay_mask(img_file, result_mask))
