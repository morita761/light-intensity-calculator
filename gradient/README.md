# gradient

マスク画像に描いた色付き長方形の中で、元画像の**緑色輝度が左 → 右にどう変化するか（グラディエント）**を
プロットするフォルダ。

## フォルダ構成

```
gradient/
├── config/
│   ├── image_sets.txt        # 画像セット（元画像 | マスク）
│   └── figure_settings.txt   # 検出色・タイトル・軸ラベル
├── data/sample/              # サンプル画像（git管理）
├── pics/                     # 入力画像（git管理外）
├── output/                   # 出力（git管理外）
├── lib/profiles.py           # 長方形の検出と輝度プロファイル
└── *.py                      # 実行スクリプト
```

## 使い方

```bash
python make_dummy_gradient.py                          # サンプル画像を作る（data/sample/ に作成済み）
python plot_gradient.py --set dummy                    # サンプルで動作確認
python plot_gradient.py --set control --colors blue yellow
python plot_gradient.py -h                             # オプション・入出力の説明
```

## スクリプト一覧

| スクリプト | 何用か | 旧スクリプト |
|---|---|---|
| `plot_gradient.py` | 長方形内の緑チャネルを列ごとに平均した輝度プロファイルをプロット | `plot-gradient.py` |
| `make_dummy_gradient.py` | 動作確認用のダミー画像（3種類のグラディエント + 青い長方形）を作る | `dumy-plot-gradient.py` |

### `plot_gradient.py`
- **モード**（`--mode`、既定 auto = 1組なら single、複数組なら multi）
  - `single`: 1組ずつ、各長方形のプロファイル（それぞれの最大値で0〜1）と、その平均
  - `multi`: 複数組の平均プロファイルを重ねる（全画像共通の最大値で0〜1）
- **主なオプション**: `--set`, `--colors`（長方形の色: blue, yellow, red, green, cyan, magenta）
- **出力**: `output/plot_gradient/<セット>_<番号>_regions.png`, `_average.png`（single）, `<セット>_multi.png`（multi）, `<セット>/rects_*.png`（検出した長方形）

長さの違うプロファイルは、最も短いものの長さに線形補間でそろえてから平均する。

## 共通オプション

`--no-text`（文字無し版のみ）/ `--both`（通常版 + 文字無し版）/ `--no-show` / `--image-sets <txt>`。
`stats_figures/` などと共通（`shared/figure.py`）。

## 設定ファイル

- `config/image_sets.txt`: 1行に `元画像 | マスク`（同じファイルでもよい）。書式は `heatmap/README.md` を参照
- `config/figure_settings.txt`: 既定のセット・検出色・面積の閾値・各図のタイトル/軸ラベル/色

## 必要なライブラリ

opencv-python, numpy, matplotlib
