# Mask R-CNNを用いた馬蹄形の検出モデル

## Detectron2

Facebookの物体検出アルゴリズム  
(https://demura.net/deeplearning/16791.html)  
python3.11にはまだ対応していないので，3.10にダウングレードする  
visual studioのC++14もインストールした  

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

