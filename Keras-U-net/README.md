# UnetモデルKerasを使用したテスト

# 環境構築
```
pip install matplotlib
pip install scikit-learn
pip install tensorflow
```

# モデルが全ピクセルを背景と予測してしまったときの対策
1. 損失関数を Dice loss に変えてみる
2. パッチ抽出時に「対象の割合」を見る
3. Augmentation を導入
4. モデル出力を可視化・中間層確認

# モデルを保存（構造＋重み＋最適化状態すべて含む）
```
model.save("unet_model.h5")
```
# 呼び出し方法（モデルの読み込み）
```
from tensorflow.keras.models import load_model
# 必要であればカスタム損失関数なども渡す
model = load_model("unet_model.h5", compile=False)

model.compile()
```
# カスタム損失関数がある場合
```
# カスタム損失関数を定義
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# load_model時に明示的に渡す
model = load_model("unet_model.h5", custom_objects={'dice_loss': dice_loss})
```