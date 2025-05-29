# Mask R-CNNを用いた馬蹄形の検出モデル

## Detectron2

Facebookの物体検出アルゴリズム  
(https://demura.net/deeplearning/16791.html)  
python3.11にはまだ対応していないので，3.10にダウングレードする  
visual studioのC++14もインストールした  
Lenovoはnvidiaがあったので，Python 3.10 & CUDA 11.8対応PyTorchをinstall  
nvidiaの確認コマンド
```
nvidia-smi
```
正常に動作した

```success
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas
```
```
pip install cython
pip install pycocotools
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install torch==1.13.1 torchvision==0.14.1
pip uninstall torchmetrics
```
## エラーが出たので対応
Python・CUDA・Torch（torch・torchvision・torchaudioの3種類ある）の3つのバージョンの整合性
```
AssertionError: Torch not compiled with CUDA enabled
```
```
pip uninstall torch
pip cache purge
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8 の例

```
メモリ不足が原因  
num_workersをエラーが出ない値まで小さくする
ex) num_workers = 0　等
Windowsでは multiprocessing を使うとき、必ず if __name__ == "__main__" を使わないとクラッシュします。
```
RuntimeError: DataLoader worker (pid(s) 5772, 15988) exited unexpectedly
```
```
project/
├── simpledataset/
│   ├── images/003.png
│   └── masks/003_mask.png
├── dataset_preparation.py  ← マスク作成＆アノテーション変換
├── train.py                ← Mask R-CNNの学習コード
├── inference.py            ← 推論コード（003.pngから馬蹄形検出）
```

## 正常動作したときのpip list(python3.10.10, CUDA対応PC)
```
Package                      Version            Editable project location
---------------------------- ------------------ -------------------------------------------------------------------------
absl-py                      2.3.0
aiohappyeyeballs             2.6.1
aiohttp                      3.12.2
aiosignal                    1.3.2
albucore                     0.0.24
albumentations               2.0.8
annotated-types              0.7.0
antlr4-python3-runtime       4.9.3
astunparse                   1.6.3
async-timeout                5.0.1
attrs                        25.3.0
black                        25.1.0
certifi                      2025.4.26
charset-normalizer           3.4.2
click                        8.2.1
cloudpickle                  3.1.1
colorama                     0.4.6
contourpy                    1.3.2
cycler                       0.12.1
Cython                       3.1.1
detectron2                   0.6                c:\users\morit\documents\light-intensity-calculator\detectron2\detectron2
filelock                     3.18.0
flatbuffers                  25.2.10
fonttools                    4.58.1
frozenlist                   1.6.0
fsspec                       2025.5.1
fvcore                       0.1.5.post20221221
gast                         0.6.0
google-pasta                 0.2.0
grpcio                       1.71.0
h5py                         3.13.0
huggingface-hub              0.32.2
hydra-core                   1.3.2
idna                         3.10
iopath                       0.1.9
Jinja2                       3.1.6
joblib                       1.5.1
keras                        3.10.0
kiwisolver                   1.4.8
libclang                     18.1.1
lightning-utilities          0.14.3
Markdown                     3.8
markdown-it-py               3.0.0
MarkupSafe                   3.0.2
matplotlib                   3.10.3
mdurl                        0.1.2
ml_dtypes                    0.5.1
mpmath                       1.3.0
multidict                    6.4.4
mypy_extensions              1.1.0
namex                        0.1.0
networkx                     3.4.2
numpy                        2.1.3
omegaconf                    2.3.0
opencv-contrib-python        4.11.0.86
opencv-python                4.11.0.86
opencv-python-headless       4.11.0.86
opt_einsum                   3.4.0
optree                       0.16.0
packaging                    25.0
pathspec                     0.12.1
pillow                       11.2.1
pip                          22.3.1
platformdirs                 4.3.8
portalocker                  3.1.1
propcache                    0.3.1
protobuf                     5.29.4
pycocotools                  2.0.8
pydantic                     2.11.5
pydantic_core                2.33.2
Pygments                     2.19.1
pyparsing                    3.2.3
python-dateutil              2.9.0.post0
pytorch-lightning            1.9.5
pywin32                      310
PyYAML                       6.0.2
requests                     2.32.3
rich                         14.0.0
safetensors                  0.5.3
scikit-learn                 1.6.1
scipy                        1.15.3
segmentation_models_pytorch  0.5.0
setuptools                   65.5.0
simsimd                      6.2.1
six                          1.17.0
stringzilla                  3.12.5
sympy                        1.13.3
tabulate                     0.9.0
tensorboard                  2.19.0
tensorboard-data-server      0.7.2
tensorflow                   2.19.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    3.1.0
threadpoolctl                3.6.0
timm                         1.0.15
tomli                        2.2.1
torch                        2.7.0+cu118
torchaudio                   2.7.0+cu118
torchmetrics                 1.7.2
torchvision                  0.22.0+cu118
tqdm                         4.67.1
typing_extensions            4.13.2
typing-inspection            0.4.1
urllib3                      2.4.0
Werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.17.2
yacs                         0.1.8
yarl                         1.20.0
```