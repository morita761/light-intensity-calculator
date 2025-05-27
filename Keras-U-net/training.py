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
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.augmentations import *

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
# --- ファイルパスの取得 ---
image_files = sorted(glob.glob("./simpledataset/images/*"))
mask_files = sorted(glob.glob("./simpledataset/masks/*"))
# image_files = sorted(glob.glob("../data/img/*.tif"))
# mask_files = sorted(glob.glob("../data/masks/*_mask.tif"))
predict_image = image_files[0]

assert len(image_files) == len(mask_files), "画像とマスクの数が一致しません"

# img_file = Path("../data/up_stronger.tif")
# mask_file = Path("../data/up_stronger_mask.tif")

img = cv2.imread(image_files[0])
# mask = blue.extract_blue_mask(mask_files[0])
height, width = img.shape[:2]

# --- Albumentations によるデータ拡張関数 ---
def augment_patch(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])
    augmented = transform(image=image, mask=mask)
    return augmented["image"], augmented["mask"]

# --- すべての画像・マスクのパッチを抽出し結合 ---
X_all, y_all = [], []
for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"画像が読み込めませんでした: {img_path}")
    mask = blue.extract_blue_mask(mask_path)
    # --- パッチ抽出 ---
    X, y = patch.extract_patches(img, mask)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 各パッチに対してデータ拡張を適用
    for i in range(len(X)):
        orig_img, orig_mask = X[i], (y[i] * 255).astype("uint8").squeeze()
        aug_img, aug_mask = augment_patch(orig_img, orig_mask)
        X_all.append(aug_img / 255.0)  # 正規化
        y_all.append(aug_mask[..., None] / 255.0)  # 形状: (H, W, 1) にして正規化


X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)
# shapeの確認
print(f"X_all shape: {X_all.shape}")  # → (N, 256, 256, 3)
print(f"y_all shape: {y_all.shape}")  # → (N, 256, 256, 1)

print("すべての画像・マスクからのパッチ数:", X_all.shape[0])

# --- 訓練/検証分割 ---
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2)

# --- モデル構築とコンパイル ---
model = K.build_unet()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

# --- EarlyStopping コールバック ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- ログファイルパス ---
log_file_path = Path("logs/pred_mean_log.txt")
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# --- 学習エポックの推移 コールバック ---
class PrintPredMeanCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(X_val[:1], verbose=0)
        # print(f"Epoch {epoch}: pred.mean() = {np.mean(pred)}")
        mean_val = np.mean(pred)
        # ログに追加出力
        log_text = (
            f"Epoch {epoch} - "
            f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f} - "
            f"pred.mean(): {mean_val:.6f}\n"
        )
        print(log_text.strip())
        with open(log_file_path, "a") as f:
            f.write(log_text)

# --- 学習（エポック数増加） ---
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          batch_size=8, # 1回の勾配更新に使うデータ数
          callbacks=[early_stop, PrintPredMeanCallback()])

# --- 推論とマスク保存 ---
result_mask = predict.predict_on_test_image(predict_image, model)
padded_mask = np.zeros((height, width), dtype=result_mask.dtype)
h, w = result_mask.shape
padded_mask[:min(height, h), :min(width, w)] = result_mask[:min(height, h), :min(width, w)]

cv2.imwrite("predicted_mask.png", padded_mask)

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

cv2.imwrite("overlay_debug.png", overlay_mask(predict_image, result_mask))
