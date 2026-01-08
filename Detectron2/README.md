# Mask R-CNNを用いた馬蹄形の検出モデル

Facebook Detectron2フレームワークを使用した馬蹄形検出・輝度測定パイプライン。

## ファイル構成

```
Detectron2/
├── train.py                    # Mask R-CNN学習スクリプト
├── dataset_preparation.py      # マスクからCOCOアノテーション生成
├── inference_ver2_heatmaps.py  # 推論 + ヒートマップ生成
├── func/
│   ├── intensityCalc.py        # PCA + K-Meansによる輝度計算
│   └── splitobje.py            # 画像分割・緑色抽出
├── simpledataset/              # サンプルデータセット
├── tensorboard/                # 学習ログ
└── *_annotations*.json         # COCOアノテーションファイル
```

## 使用方法

```bash
# 1. データセット準備（マスク画像からアノテーション生成）
python dataset_preparation.py

# 2. モデル学習
python train.py

# 3. 推論 + ヒートマップ生成
python inference_ver2_heatmaps.py
```

## 各ファイルの説明

### train.py
Mask R-CNN（ResNet-50 FPN）の学習スクリプト。
- COCOアノテーション形式で学習
- 学習済みモデルは `./output/` に保存

### dataset_preparation.py
マスク画像からCOCO形式のアノテーションJSONを生成。
- `simpledataset/images/` - 入力画像
- `simpledataset/masks/` - マスク画像

### inference_ver2_heatmaps.py
推論と可視化を行うメインスクリプト。
- 馬蹄形検出
- V-D方向（左右）に分割
- ヒートマップ生成
- 箱ひげ図による統計表示

### func/intensityCalc.py
輝度計算の共通関数。
- `extract_green_points()` - 緑色領域をHSVで検出
- `perform_pca()` - PCAで主成分方向を計算
- `perform_kmeans()` - K-Meansで左右クラスタリング
- `draw_results()` - 結果の描画

### func/splitobje.py
画像分割の共通関数。
- `extract_green_object()` - 指定範囲外をマスク化
- `split_left_right()` - 緑色領域の中心で左右分割

## 環境構築

```bash
# PyTorch（CUDA 11.8対応）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Detectron2
pip install cython pycocotools
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install -e .

# その他
pip install pandas scikit-learn matplotlib seaborn
```

## 動作確認済み環境

- Python 3.10
- PyTorch 2.7.0+cu118
- Detectron2 0.6
- CUDA 11.8

## トラブルシューティング

### CUDA未有効エラー
```
AssertionError: Torch not compiled with CUDA enabled
```
→ CUDA対応のPyTorchを再インストール

### メモリ不足
```
RuntimeError: DataLoader worker exited unexpectedly
```
→ `train.py` の `num_workers` を小さくする（例: 0）
