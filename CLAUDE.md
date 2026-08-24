# CLAUDE.md

このファイルはClaude Codeがこのリポジトリを操作する際のガイダンスを提供します。

## プロジェクト概要

画像から馬蹄形（horseshoe）領域を検出し、V-D方向（Ventral/Dorsal）に分割して光強度（輝度）を測定するプロジェクト。

## ファイル検索パターン

ファイルを検索する際は、以下のパターンを使用してプロジェクト全体から検索すること：

```
# ファイル名で検索（推奨）
**/ファイル名.py

# 例
**/main.py
**/gptFunc.py
**/intensityCalc.py
```

## フォルダ構成

```
light-intensity-calculator/
├── Detectron2/           # 深層学習による馬蹄形検出（メイン）
│   ├── train.py          # Mask R-CNN学習スクリプト
│   ├── inference_*.py    # 推論スクリプト群
│   └── func/             # 輝度計算・分割処理
├── Cellpose/             # Cellposeによる馬蹄形検出（検証中）
│   ├── inference_test.py # 事前学習モデルのゼロショット検出テスト
│   ├── dataset_prepare_from_coco.py # 既存COCOアノテーションの変換
│   ├── train_finetune.py # fine-tuningスクリプト
│   └── func/              # V-D分割・輝度計算
├── horseShoeShapeRecog/  # OpenCVによる従来手法（実験）
├── test-heatmap/         # ヒートマップ可視化
├── test_fig/             # 統計グラフ（箱ひげ図等）
├── test-histogram/       # ヒストグラム処理テスト
├── test-gradient/        # グラデーション処理テスト
├── test-circle/          # 円検出テスト
├── test_arrow/           # 矢印検出・極座標ヒストグラム
├── pic/                  # テンプレート画像
├── archive/              # 過去の実験コード（Keras/PyTorch U-Net）
└── ルートファイル         # 従来手法のメインスクリプト
```

## 主要ファイル

### Detectron2（メインパイプライン）
- `**/train.py` - Mask R-CNN学習
- `**/inference_ver2_heatmaps.py` - 改良版ヒートマップ推論
- `**/intensityCalc.py` - PCA + K-Meansによる左右分割・輝度計算
- `**/splitobje.py` - 馬蹄形の分離処理

### Cellpose（検証中の代替パイプライン、詳細は Cellpose/README.md）
- `**/inference_test.py` - Cellpose-SAM事前学習/fine-tuning済みモデルによる検出テスト（diameter/flow_threshold/cellprob_threshold指定可）
- `**/estimate_diameter.py` - COCOアノテーションから推奨diameterを算出
- `**/dataset_prepare_from_coco.py` - Detectron2用COCOアノテーションをCellpose形式に変換
- `**/train_finetune.py` - 変換済みデータでのCLI fine-tuning（loss履歴CSV・TensorBoardログ出力付き）
- `**/launch_gui.py` - Cellpose GUI起動（human-in-the-loop fine-tuning用、実行・検証はユーザー側）

### 可視化・統計
- `**/boxplot_vd.py` - V-D分岐の箱ひげ図比較（t検定付き）
- `**/mosaic_horseshoe.py` - モザイク実験用ヒートマップ
- `**/orientation-polar-histogram.py` - 極座標ヒストグラム

### ルートレベル
- `**/gptFunc.py` - 緑色抽出・左右分割の共通関数
- `**/brightnessMeasure.py` - 輝度測定

## コマンド

```bash
# 学習
python Detectron2/train.py

# 推論（輝度計算）
python Detectron2/inference_split_calc.py

# 推論（ヒートマップ付き）
python Detectron2/inference_heatmaps.py
```

## 開発ガイドライン

- 言語: Python
- 主要ライブラリ: OpenCV, Detectron2, PyTorch, matplotlib, seaborn, scikit-learn, scipy
- 画像処理: OpenCV（cv2）を使用
- 深層学習: Detectron2（Mask R-CNN）を使用
- 統計処理: scipy.stats（t検定等）を使用

## 注意事項

- `archive/` フォルダは過去の実験コードであり、現在は使用されていない
- メインの処理パイプラインは `Detectron2/` フォルダにある
- 新しい機能追加時は既存のコーディングスタイルに従うこと
