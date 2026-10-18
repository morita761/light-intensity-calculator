# heatmap

馬蹄形を同じ大きさにそろえて重ね合わせ、**馬蹄形のどこが明るいか（GFP輝度の偏り）**を
Ventral / Dorsal 別の平均ヒートマップで可視化するフォルダ。

## フォルダ構成

```
heatmap/
├── config/
│   ├── image_sets.txt        # 画像セット（どの画像を使うか）
│   └── figure_settings.txt   # タイトル・カラーマップ・解析パラメータ（スクリプトごと）
├── pics/                     # 入力画像（git管理外。image_sets.txt の dir の置き場所）
├── output/                   # 出力（スクリプト名ごと、git管理外）
├── lib/                      # 共通処理（ROI抽出・平均化・作図部品）
├── legacy/                   # 初期の単一画像版（現在は未使用）
└── *.py                      # 実行スクリプト
```

## 使い方

```bash
python horseshoe_heatmap.py --set vang_cas9            # 1セットの平均ヒートマップ
python group_heatmap.py --groups control vang_rnai vang_cas9   # 系統間を同じスケールで比較
python mosaic_compare.py --set mosaic_control          # 通常マスク vs RFPマスク
python horseshoe_heatmap.py -h                         # オプション・入出力の説明
```

細かいオプションは **すべて `python <スクリプト名> -h`** で確認できる。

## スクリプト一覧

| スクリプト | 何用か | 旧スクリプト |
|---|---|---|
| `horseshoe_heatmap.py` | 1つの画像セットの平均ヒートマップ（左右それぞれ最小〜最大を0〜255に） | `horseshoe-heatmaps.py`, `trim_test.py`（`--trim`）, `mosaic_horseshoe.py`（`--overlay-rfp --histogram`） |
| `group_heatmap.py` | 複数系統の平均ヒートマップを全系統共通の最大値でそろえて比較 + 輝度分布ヒストグラム | `show-intensity-histgram.py`, `show-intensity-histgram-three-groups.py` |
| `mosaic_compare.py` | モザイク個体で、通常マスクと RFPマスク（遺伝子操作した細胞）の平均ヒートマップを比較 | `test_mosaic.py` |

### `horseshoe_heatmap.py`
- **用途**: 画像セット内の全馬蹄形を 100x100 にそろえて画素ごとに平均し、Ventral / Dorsal の輝度分布を見る
- **主なオプション**: `--set`, `--trim N`（トリム平均）, `--center-margin`（中心線付近の除外幅）, `--overlay-rfp`（RFP画像上にアノテーション）, `--histogram`
- **出力**: `output/horseshoe_heatmap/<セット>_overview.png`（アノテーション + 左右）, `_ventral.png`, `_dorsal.png`, `<セット>/annotated_*.png`

```bash
python horseshoe_heatmap.py --set vang_cas9
python horseshoe_heatmap.py --set vang_cas9 --trim 1 --center-margin 50
python horseshoe_heatmap.py --set mosaic_fz --overlay-rfp --histogram
```

### `group_heatmap.py`
- **用途**: Control / vang RNAi / vang Cas9 などを、Global Mean を全系統で共通にし、全系統・左右共通の最大値を255にして比較する（系統間で明るさを比べられる）
- **主なオプション**: `--groups`（並べる順）, `--hist-source raw|average`（ヒストグラムに使う値）, `--center-margin`
- **出力**: `output/group_heatmap/<グループ>_<セット>.png`, `<グループ>_histogram.png`

```bash
python group_heatmap.py --groups control vang_rnai vang_cas9
python group_heatmap.py --groups control vang_cas9 --hist-source average
```

### `mosaic_compare.py`
- **用途**: 通常マスク（全馬蹄形）と RFP マスク（RFP陽性の馬蹄形）の GFP 輝度分布を比べる。明るさの補正と 0〜255 のスケールは通常マスク基準
- **入力**: image_sets.txt に `元画像 | 通常マスク | RFP画像 | RFPマスク` の4列で書いたセット
- **出力**: `output/mosaic_compare/<セット>_overview.png`, `<セット>_grid.png`（2行2列 + 共通カラーバー）

```bash
python mosaic_compare.py --set mosaic_control
python mosaic_compare.py --set mosaic_fz --both --no-show
```

## 共通オプション

```bash
python horseshoe_heatmap.py --set control --no-text     # 文字無し版のみ（*_notext.png）
python horseshoe_heatmap.py --set control --both        # 通常版 + 文字無し版
python horseshoe_heatmap.py --set control --no-show     # 画面表示なしで保存だけ
python horseshoe_heatmap.py --set test --image-sets my_sets.txt   # 別の画像セット定義を使う
```

文字無し版は通常版と同じ範囲で切り出すので、Illustrator 等で重ねると位置が一致する
（`stats_figures/` と共通の `shared/figure.py`）。

## 設定ファイル

### `config/image_sets.txt`（使う画像）

```ini
[control]
label = Control                     ; グラフに表示する名前
dir = control                       ; pics/ からのフォルダ
images =
    Projections of 241106_Fz_002_second.png          ← マスクは「元画像名 + _mask.tif」
    元画像.png | マスク.tif                            ← マスク名が違う場合
    元画像.png | マスク.tif | RFP画像.png | RFPマスク.tif   ← mosaic_compare.py 用
```

（実際のファイルでは `;` のコメントは書けない。コメントは行頭 `#`）

### `config/figure_settings.txt`（図と解析パラメータ）

- `[common]`: dpi・保存形式・カラーマップ・**ROI抽出パラメータ**（`center_margin`, `dilate_occupancy`, `max_dilate`, `clip_uint8`, `target_size`, `annotation_colors`）
- `[スクリプト名]`: タイトル・図サイズ・既定のセット・スクリプト固有の値。旧スクリプトと同じ結果になるよう、`group_heatmap` は `center_margin = 50`, 膨張なし, 小数のまま平均に設定済み

## 処理の流れ（全スクリプト共通、`lib/core.py`）

1. 全画像のマスク内 GFP 輝度の平均 = **Global Mean**
2. 画像ごとに `scaling = Global Mean / その画像のマスク内平均` で明るさをそろえる
3. マスクの各馬蹄形を、画像中心線の左（Ventral）/ 右（Dorsal）に分類（中心線をまたぐもの・`center_margin` 以内のものは除外）
4. 馬蹄形のバウンディングボックスの GFP 輝度（馬蹄形の外は0）を 100x100 にリサイズして集め、画素ごとに平均

マスク画像は、元画像の上に馬蹄形を blue / cyan / magenta で塗ったもの。

## legacy/

初期の単一画像・切り抜き版（`heatmap.py`, `heatmap_multiple.py`, `color-horseshoe-heatmap.py`,
`create_dummy_and_heatmap.py`, `indivisual-heatmap.py`）。現在のスクリプトに置き換え済みで、参照用に残している。

## 必要なライブラリ

opencv-python, numpy, matplotlib
