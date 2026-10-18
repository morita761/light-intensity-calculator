# arrow_analysis

馬蹄形の**向き（凹 = 開口部が向いている方向）**を推定し、Ventral / Dorsal 別の角度分布を
極座標ヒストグラムで表示・2群比較するフォルダ。

角度の定義: **画像の下 = 0°、右 = 90°、左 = -90°、上 = ±180°**

## フォルダ構成

```
arrow_analysis/
├── config/
│   ├── image_sets.txt        # 画像セット（どの画像を使うか）
│   └── figure_settings.txt   # タイトル・色・推定パラメータ（スクリプトごと）
├── pics/                     # 入力画像（git管理外）
├── output/                   # 出力（スクリプト名ごと、git管理外）
├── lib/                      # 向きの推定（orientation.py）・作図部品（plots.py）
├── legacy/                   # 極座標グラフの見た目の試作（simple-polar-histogram.py）
└── *.py                      # 実行スクリプト
```

## 使い方

```bash
python orientation_polar.py --set R8control                    # 1セットの向きの分布
python group_comparison_ttest.py --control R8control --test R8ori   # 2群の t検定
python orientation_polar.py -h                                 # オプション・入出力の説明
```

細かいオプションは **すべて `python <スクリプト名> -h`** で確認できる。

## スクリプト一覧

| スクリプト | 何用か | 旧スクリプト |
|---|---|---|
| `orientation_polar.py` | 1つの画像セットの向きを推定し、矢印を描いた画像と Ventral / Dorsal の極座標ヒストグラムを出力 | `detect_arrow_improved.py`, `detect_arrow.py`, `orientation-polar-histogram.py`（`--method moment`） |
| `group_comparison_ttest.py` | Control 群と Test 群で向きを Welch の t検定で比較（箱ひげ図 + 極座標ヒストグラム） | `group_comparison_ttest.py` |

### `orientation_polar.py`
- **用途**: 各馬蹄形の開口部の向きを推定し、角度分布を見る
- **主なオプション**:
  - `--method skeleton`（既定）: 骨格化して両端点の中点を開口部とし、凸包の重心 → 端点中点 の向き
  - `--method moment`: 輪郭の重心 → バウンディングボックス中心 の向き（旧方式）
  - `--count`: 縦軸を割合でなく個数に / `--center-margin`: 中心線付近の除外幅
  - `--debug-rois`: 馬蹄形ごとの拡大画像 + 矢印 / `--debug-skeleton`: 骨格・端点（緑）・分岐点（赤）
- **出力**: `output/orientation_polar/<セット>_<方法>_polar.png`, `_angles.csv`（全角度）, `<セット>/annotated_*.png`

```bash
python orientation_polar.py --set R8control
python orientation_polar.py --set R8ori --method moment --center-margin 50
python orientation_polar.py --set R8ori --debug-rois --debug-skeleton --no-show
```

### `group_comparison_ttest.py`
- **用途**: 2群（例: Control vs vang Cas9）で開口部の向きが変わるかを Ventral / Dorsal 別に Welch の t検定
- **主なオプション**: `--control`, `--test`, `--method`, `--center-margin`
- **出力**: `output/group_comparison_ttest/<control>_vs_<test>.png`（上段: 箱ひげ図、下段: 極座標ヒストグラム）, `_ttest.csv`

```bash
python group_comparison_ttest.py --control R8control --test R8ori
python group_comparison_ttest.py --control 29d_R8control --test 29d_R8ori
```

## 共通オプション

`--no-text`（文字無し版のみ）/ `--both`（通常版 + 文字無し版）/ `--no-show` / `--image-sets <txt>`。
`stats_figures/`, `heatmap/` と共通（`shared/figure.py`）。

## 設定ファイル

- `config/image_sets.txt`: 書式は `heatmap/README.md` と同じ（`元画像` だけ書くとマスクは `元画像名 + _mask.tif`）
- `config/figure_settings.txt`:
  - `[common]`: `method`, `center_margin`（既定 80px）, `bin_width_deg`（ヒストグラムのビン幅、既定 10°）, アノテーションの矢印・塗り
  - `[orientation_polar]` / `[group_comparison_ttest]`: 既定のセット、タイトル、色、箱ひげ図の軸範囲など

## 必要なライブラリ

opencv-python, numpy, matplotlib, scipy, scikit-image（skeleton 法）
