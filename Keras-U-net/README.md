# UnetモデルKerasを使用したテスト

# 環境構築
```
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install albumentations
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

```
Package                      Version
---------------------------- -----------
absl-py                      2.2.2
aiohappyeyeballs             2.6.1
aiohttp                      3.11.18
aiosignal                    1.3.2
albucore                     0.0.24
albumentations               2.0.7
annotated-types              0.7.0
astunparse                   1.6.3
attrs                        25.3.0
certifi                      2025.4.26
charset-normalizer           3.4.2
colorama                     0.4.6
contourpy                    1.3.0
cycler                       0.12.1
filelock                     3.18.0
flatbuffers                  25.2.10
fonttools                    4.54.1
frozenlist                   1.6.0
fsspec                       2025.5.0
gast                         0.6.0
google-pasta                 0.2.0
grpcio                       1.71.0
h5py                         3.13.0
huggingface-hub              0.31.4
idna                         3.10
Jinja2                       3.1.6
joblib                       1.5.1
keras                        3.10.0
kiwisolver                   1.4.7
libclang                     18.1.1
lightning-utilities          0.14.3
Markdown                     3.8
markdown-it-py               3.0.0
MarkupSafe                   3.0.2
matplotlib                   3.9.2
mdurl                        0.1.2
ml_dtypes                    0.5.1
mpmath                       1.3.0
multidict                    6.4.4
namex                        0.0.9
networkx                     3.4.2
numpy                        2.1.2
opencv-contrib-python        4.11.0.86
opencv-python                4.11.0.86
opencv-python-headless       4.11.0.86 // GUIなし版
opt_einsum                   3.4.0
optree                       0.15.0
packaging                    24.1
pillow                       10.4.0
pip                          22.3.1
propcache                    0.3.1
protobuf                     5.29.4
pydantic                     2.11.5
pydantic_core                2.33.2
Pygments                     2.19.1
pyparsing                    3.2.0
pyserial                     3.5
python-dateutil              2.9.0.post0
pytorch-lightning            1.9.5
PyYAML                       6.0.2
requests                     2.32.3
rich                         14.0.0
safetensors                  0.5.3
scikit-learn                 1.6.1
scipy                        1.15.3
segmentation_models_pytorch  0.5.0
setuptools                   65.5.0
simsimd                      6.2.1
six                          1.16.0
stringzilla                  3.12.5
sympy                        1.14.0
tensorboard                  2.19.0
tensorboard-data-server      0.7.2
tensorflow                   2.19.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    3.1.0
threadpoolctl                3.6.0
timm                         1.0.15
torch                        2.7.0
torchmetrics                 1.7.1
torchvision                  0.22.0
tqdm                         4.67.1
typing_extensions            4.13.2
typing-inspection            0.4.1
urllib3                      2.4.0
Werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.17.2
yarl                         1.20.0
```

pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84
pip install opencv-python-headless==4.10.0.84