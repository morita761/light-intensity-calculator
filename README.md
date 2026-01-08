# Light Intensity Calculator

画像から馬蹄形（horseshoe）領域を検出し、V-D方向（Ventral/Dorsal）に分割して光強度（輝度）を測定するプロジェクト。

## フォルダ構成

```
light-intensity-calculator/
├── Detectron2/           # 深層学習による馬蹄形検出（メイン）
├── horseShoeShapeRecog/  # OpenCVによる従来手法（実験）
├── test-heatmap/         # ヒートマップ可視化
├── test_fig/             # 統計グラフ（箱ひげ図等）
├── test-histogram/       # ヒストグラム処理テスト
├── test-gradient/        # グラデーション処理テスト
├── test-circle/          # 円検出テスト
├── test_arrow/           # 矢印検出テスト
├── pic/                  # テンプレート画像
├── archive/              # 過去の実験コード
└── ルートファイル         # 従来手法のメインスクリプト
```

---

## 各フォルダの説明

### Detectron2/（メイン）
Facebook Detectron2フレームワークを使用した物体検出・セグメンテーション。**現在の主要なパイプライン**。

| ファイル | 説明 |
|---------|------|
| `train.py` | Mask R-CNN（ResNet-50 FPN）の学習スクリプト |
| `inference_split_calc.py` | 推論 + V-D分割 + 輝度計算（CSV出力） |
| `inference_heatmaps.py` | 推論 + ヒートマップ生成（複数画像対応） |
| `inference_ver2_heatmaps.py` | 改良版ヒートマップ（箱ひげ図付き） |
| `inference_threshold.py` | 信頼度閾値によるフィルタリング推論 |
| `func/intensityCalc.py` | PCA + K-Meansによる左右分割・輝度計算 |
| `func/splitobje.py` | 馬蹄形の分離処理 |

**使い方:**
```bash
# 学習
python Detectron2/train.py

# 推論（輝度計算）
python Detectron2/inference_split_calc.py

# 推論（ヒートマップ付き）
python Detectron2/inference_heatmaps.py
```

---

### test-heatmap/
馬蹄形の輝度分布をヒートマップで可視化。Seaborn/matplotlibを使用。

| ファイル | 説明 |
|---------|------|
| `mosaic_horseshoe.py` | モザイク実験用ヒートマップ（最新） |
| `horseshoe-heatmaps.py` | 馬蹄形のヒートマップ可視化 |
| `heatmap.py` | 基本的なヒートマップ生成 |
| `color-horseshoe-heatmap.py` | 色付きヒートマップ |

---

### test_fig/
統計解析用のグラフ生成。Ventral/Dorsal間の比較分析。

| ファイル | 説明 |
|---------|------|
| `boxplot_vd.py` | V-D分岐の箱ひげ図比較（t検定付き） |
| `process_data.py` | データ前処理・正規化 |
| `linefig.py` | 線グラフ描画 |
| `outputfig.py` | 図表出力 |

---

### horseShoeShapeRecog/
OpenCVを使用した従来的な画像処理手法。Detectron2導入前の実験コード。

- CLAHE（局所適応ヒストグラム平坦化）
- 適応的閾値処理
- モルフォロジー処理
- 輪郭抽出 + 形状フィルター

---

### test-histogram/, test-gradient/, test-circle/, test_arrow/
各種画像処理手法のテスト・実験用フォルダ。

---

### archive/
過去の実験コード（Keras U-Net, PyTorch U-Net）。Detectron2が主流になったため、参照用として保存。

---

## ルートレベルのファイル（従来手法）

| ファイル | 説明 |
|---------|------|
| `main.py` | 従来手法のメインパイプライン |
| `gptFunc.py` | 緑色抽出・左右分割の共通関数 |
| `brightnessMeasure.py` | 輝度測定（円形領域マスク） |
| `matchesTemplate.py` | テンプレートマッチング |
| `tempHorseshoe.py` | テンプレート馬蹄形定義 |

---

## 環境構築

```bash
# OpenCV
pip install opencv-python opencv-contrib-python

# PyTorch + Detectron2（CUDA対応）
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# 可視化
pip install matplotlib seaborn

# 機械学習
pip install scikit-learn pandas numpy scipy
```

---

## 処理パイプライン（従来手法 - 参考）

1. **画像からメダラを抽出** → 緑の領域と画像のトリミング
2. **メダラをV-D方向に分ける** → `cv2.inRange`による緑マスク化から中央値を計算し左右分割
3. **馬蹄形を認識** → gray scale, コントラスト補正
4. **輝度測定**

---

## 技術メモ

<details>
<summary>OpenCV関連の技術メモ（クリックで展開）</summary>

### ヒストグラム平坦化（CLAHE）
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)
```

### 適応的しきい値
```python
adaptive = cv2.adaptiveThreshold(
    equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, blockSize=11, C=2)
```

### モルフォロジー処理
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
```

### Canny法（エッジ検出）
```python
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
```

### 輪郭抽出 + フィルタ
```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 300 < area < 1000:
        cv2.drawContours(mask, [cnt], -1, 255, -1)
```

</details>
