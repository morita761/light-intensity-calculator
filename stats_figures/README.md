# stats_figures

Ventral/Dorsal間の輝度比較の統計解析・グラフ作成フォルダ。

## フォルダ構成

```
stats_figures/
├── config/
│   ├── figure_settings.txt   # 図の軸・タイトル・軸範囲・サイズ・色（スクリプトごとのセクション）
│   └── datasets.txt          # 比較する系統の定義（表示名 = 生データファイル）
├── data/
│   ├── raw/                  # 入力: 輝度測定結果のテキスト（control.txt など）
│   └── processed/            # 中間: ノーマライズ済みCSV
│       ├── two_groups/       #   parse_intensity.py が datasets.txt の各セクションごとに生成
│       ├── three_groups/
│       └── single/           #   process_data.py の出力（boxplot_vd.py 用）
├── output/                   # 出力: 図（スクリプト名ごとのサブフォルダ、git管理外）
│   └── legacy/               #   リファクタリング前に生成された図
├── lib/                      # 共通モジュール（パス・データ入出力・統計。設定読込と図の保存はリポジトリ直下の shared/）
└── *.py                      # 実行スクリプト
```

## 処理の流れ

```
data/raw/*.txt
   │  python parse_intensity.py           （画像ごとにノーマライズ、系統・V/D別に分割）
   ▼
data/processed/<dataset>/*_v.csv, *_d.csv
   │  python violin_plot.py --dataset <dataset> など
   ▼
output/<スクリプト名>/*.png
```

生データを差し替えたら、まず `python parse_intensity.py` を実行し直すこと。

## 使い方（クイックスタート）

stats_figures フォルダで実行する（どのフォルダから実行してもパスは自動で解決される）。

```bash
python parse_intensity.py                          # 1. 生データ → CSV（最初に1回、生データを変えたら再実行）
python violin_plot.py --both                       # 2. 好きな図を作る
python violin_plot.py -h                           # 各スクリプトの詳しいオプション・説明
```

細かいオプション・入出力・用途は **すべて `python <スクリプト名> -h` で確認できる**。

## スクリプト一覧

| スクリプト | 何用か | 起動例 |
|---|---|---|
| `parse_intensity.py` | **前処理**。生データをノーマライズして系統×V/D別CSVに分割 | `python parse_intensity.py` |
| `process_data.py` | 前処理（旧方式、1ファイルのみ）。`boxplot_vd.py` 用 | `python process_data.py` |
| `violin_plot.py` | 極性の強さ \|L - R\| を系統間で比較（split violin） | `python violin_plot.py --dataset three_groups` |
| `absolute_diff.py` | V/Dまとめた \|L - R\| の系統間比較（棒グラフ） | `python absolute_diff.py` |
| `dot_interaction_absolute.py` | 左右branch輝度の個別ドット+平均 と \|L - R\| 棒グラフ | `python dot_interaction_absolute.py` |
| `outputfig.py` | 系統×V/Dごとの Ventral/Dorsal branch 箱ひげ図（1枚ずつ、論文用） | `python outputfig.py` |
| `boxplot_vd.py` | 1系統の V-D分岐箱ひげ図を2枚横並び | `python boxplot_vd.py` |
| `twoway_anova.py` | KOで branch間差が緩和されるかの Two-way ANOVA（詳細版、図4種） | `python twoway_anova.py` |
| `twoway_anova_genotype_x_branch.py` | Side別 Two-way ANOVA + Tukey、結果を表画像でも保存 | `python twoway_anova_genotype_x_branch.py` |
| `linefig.py` | 複数CSVの最小〜最大+平均を並べる汎用グラフ | `python linefig.py --files a.csv b.csv` |

`--dataset` の既定は `two_groups`（Control vs Cas9）。`three_groups` で Control / vang RNAi / vang Cas9。

### 前処理

#### `parse_intensity.py`
- **用途**: 全作図スクリプトの前処理。輝度測定結果（`data/raw/*.txt`）を画像ごとの明るさの違いを補正してCSVにする
- **入力**: `data/raw/*.txt`（どのファイルを使うかは `config/datasets.txt`）
- **出力**: `data/processed/<dataset>/<ファイル名>_v.csv`, `_d.csv`

```bash
python parse_intensity.py                        # datasets.txt の全セクション
python parse_intensity.py --dataset two_groups   # 指定セクションのみ
```

#### `process_data.py`
- **用途**: `boxplot_vd.py` 用の前処理（1ファイルだけを扱う旧ノーマライズ方式）
- **入力**: `data/raw/control.txt`（`--input` で変更可）
- **出力**: `data/processed/single/v.csv`, `d.csv`, `v_normalize.csv`, `d_normalize.csv`

```bash
python process_data.py
python process_data.py --input vang_rnai.txt
```

### 極性（|L - R|）の比較

#### `violin_plot.py`
- **用途**: 左右branchの輝度差の絶対値（= 極性の強さ）が系統間で変わるかを見る。左半分が Ventral R8、右半分が Dorsal R8。赤線 = 中央値、点線 = 四分位。先頭の系統と他の各系統を t検定
- **出力**: `output/violin_plot/violin_plot_<dataset>.png`

```bash
python violin_plot.py
python violin_plot.py --dataset three_groups --both
```

#### `absolute_diff.py`
- **用途**: V/D を区別せず、M1層全体として極性が弱まっているかを見る（棒 = 平均、エラーバー = 95%CI）。t検定のP値はコンソールに表示
- **出力**: `output/absolute_diff/absolute_diff_<dataset>.png`

```bash
python absolute_diff.py
python absolute_diff.py --dataset three_groups
```

#### `dot_interaction_absolute.py`
- **用途**: 左パネル = Ventral/Dorsal branch の輝度（個別ドット + 平均±SE、系統で傾きが変わるか）、右パネル = |L - R| の平均±SE（個別ドット付き）
- **出力**: `output/dot_interaction_absolute/dot_interaction_absolute_<dataset>.png`

```bash
python dot_interaction_absolute.py
python dot_interaction_absolute.py --dataset three_groups
```

### Ventral branch vs Dorsal branch の箱ひげ図

#### `outputfig.py`
- **用途**: 各系統・各R8（V/D）で Ventral branch と Dorsal branch の輝度に差があるかを t検定付き白黒箱ひげ図で示す（1枚ずつ出力、論文用）
- **出力**: `output/outputfig/outputfig_<dataset>_<系統>_<V|D>.png`

```bash
python outputfig.py
python outputfig.py --dataset three_groups --both --no-show
```

#### `boxplot_vd.py`
- **用途**: 1系統について Ventral R8s / Dorsal R8s の箱ひげ図を横並びで表示（`process_data.py` の出力を使用）
- **出力**: `output/boxplot_vd/boxplot_vd.png`

```bash
python process_data.py
python boxplot_vd.py
```

### Two-way ANOVA

statsmodels が必要（`pip install statsmodels`）。

#### `twoway_anova.py`
- **用途**: 「KOで Ventral/Dorsal branch 間の輝度差が緩和されるか」を検定する詳細版。記述統計・条件別t検定・全体/Side別ANOVA（η²付き）・Tukey HSD をコンソール出力
- **出力**: `output/twoway_anova/twoway_anova_<dataset>_*.png`（交互作用 / 条件別箱ひげ図 / branch差 / まとめ）

```bash
python twoway_anova.py
python twoway_anova.py --dataset three_groups --no-show
```

#### `twoway_anova_genotype_x_branch.py`
- **用途**: Ventral R8 と Dorsal R8 を分けて 系統×部位 のANOVA と Tukey HSD（系統間の多重比較）を行い、結果を表画像でも保存する（発表・報告用、`twoway_anova.py` より簡潔）
- **出力**: `output/twoway_anova_genotype_x_branch/..._<dataset>_*.png`（ANOVA表 / Tukey表 / 交互作用 / 箱ひげ図）

```bash
python twoway_anova_genotype_x_branch.py
python twoway_anova_genotype_x_branch.py --dataset three_groups
```

### その他

#### `linefig.py`
- **用途**: 列 = カテゴリ（例: C1, BA, C2）のCSVを複数並べ、値の範囲（最小〜最大）と平均を色分けして比較する汎用グラフ。`--files` 省略でサンプルデータ
- **出力**: `output/linefig/linefig.png`

```bash
python linefig.py
python linefig.py --files a.csv b.csv c.csv
```

## 共通オプション（作図スクリプト全て）

```bash
python violin_plot.py                          # 通常版を保存して表示
python violin_plot.py --no-text                # 文字無し版のみ（*_notext.png）
python violin_plot.py --both                   # 通常版 + 文字無し版
python violin_plot.py --dataset three_groups   # 系統セットを切り替え
python violin_plot.py --both --no-show         # 画面表示なしで保存だけ
```

**文字無し版**: タイトル・軸ラベル・目盛りの数字・凡例・アスタリスク・n数などの文字を全て消し、
軸線・目盛り線・プロット・有意差ブラケットの線だけを残した図。通常版と同じ範囲で切り出すので、
Illustrator等で重ねると位置が一致する。ANOVA/Tukeyの表画像は文字無し版を作らない。

## 設定ファイル

### `config/figure_settings.txt`（軸・タイトルなど）

スクリプト名のセクションに、タイトル・軸ラベル・軸範囲・図サイズ・色などをまとめている。
`[common]` は全スクリプトの既定値（dpi・保存形式・フォントサイズ・branch名）。

```ini
[violin_plot]
figsize = 10, 7
title = Polarity Magnitude (|L - R|) Comparison\nSplit Violin Plot (Ventral vs Dorsal)
xlabel = Genotype
ylabel = Absolute Intensity Difference
# 0, 60 のように固定もできる（行末コメントは不可。コメントは行頭 # で書く）
ylim = auto
```

- `\n` で改行、値を空（`title =`）にするとその文字を表示しない
- 複数パネルの図は `ventral_title`, `left_ylabel` のように接頭辞付きキーで個別指定
- `format = pdf` / `svg` にするとベクター形式で保存

### `config/datasets.txt`（系統の定義）

```ini
[three_groups]
Control = control.txt
vang RNAi = vang_rnai.txt
vang Cas9 = loco_vang_cas9.txt
```

- 左辺がグラフ上の表示名（記載順に並ぶ、先頭が統計比較の基準）、右辺が `data/raw/` のファイル名
- セクションを追加すれば `--dataset <セクション名>` で使える（追加後 `parse_intensity.py` を実行）

## ノーマライズ方式

`parse_intensity.py`:

```
Normalized Ave = Raw Ave * (Grand Median / Image Med Average)
```

- **Grand Median**: データセット内の全 left_med / right_med の中央値
- **Image Med Average**: 各画像内（V/D両方）の left_med / right_med の平均値

Grand Median はデータセット（datasets.txt のセクション）ごとに計算されるため、
同じ control.txt でも `two_groups` と `three_groups` で値がわずかに異なる。

`process_data.py`（旧方式）:

```
Normalized Ave = Ave / (画像内 left_med平均 + right_med平均) * (全体 left_med平均 + right_med平均)
```

## 入力テキストの書式

```
- <画像タイトル>
V
1, <left_ave>, <left_med>, <right_ave>, <right_med>,
2, ...
D
1, ...
```

## 統計検定と有意性マーク

| 記号 | p値 |
|------|-----|
| ***** | p < 0.00001（boxplot_vd / outputfig のみ、`max_stars = 5`） |
| **** | p < 0.0001 |
| *** | p < 0.001 |
| ** | p < 0.01 |
| * | p < 0.05 |
| n.s. | p >= 0.05 |

## 必要なライブラリ

pandas, numpy, matplotlib, seaborn, scipy, statsmodels（ANOVA系のみ）
