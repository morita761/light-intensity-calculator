# test_fig

統計解析用のグラフ生成フォルダ。Ventral/Dorsal間の輝度比較分析を行う。

## ファイル構成

### Pythonスクリプト

| ファイル | 説明 |
|---------|------|
| `boxplot_vd.py` | V-D分岐の箱ひげ図を2つ横並びで比較表示（t検定付き） |
| `outputfig.py` | 箱ひげ図を個別に表示（control/cas9の4種類をプロット） |
| `process_data.py` | テキストファイルをV/Dで分割・ノーマライズしてCSV出力 |
| `parse_intensity_text_file.py` | control/cas9の2ファイルを画像ごとにノーマライズして4つのCSVに分割出力 |
| `linefig.py` | 複数CSVから線グラフ描画（最大/最小/平均値表示） |

### データファイル

| ファイル | 説明 |
|---------|------|
| `input_test.txt` | process_data.py用の入力テストデータ |
| `control.txt` | コントロール群の輝度データ |
| `loco_vang_cas9.txt` | Cas9群の輝度データ |
| `v.csv`, `d.csv` | V/D分割後の元データ |
| `v_normalize.csv`, `d_normalize.csv` | ノーマライズ済みV/Dデータ |
| `control_v.csv`, `control_d.csv` | コントロール群のV/D別データ |
| `cas9_v.csv`, `cas9_d.csv` | Cas9群のV/D別データ |

## 使い方

```bash
# テキストファイルからV/Dデータを抽出・ノーマライズ
python process_data.py

# control/cas9の比較用データを生成
python parse_intensity_text_file.py

# 箱ひげ図の表示（2つ横並び）
python boxplot_vd.py

# 箱ひげ図の表示（個別、control/cas9の4種類）
python outputfig.py
```

## データ処理フロー

```
入力テキストファイル (control.txt, loco_vang_cas9.txt)
    ↓
parse_intensity_text_file.py
    ↓ (画像ごとのノーマライズ)
4つのCSV (control_v.csv, control_d.csv, cas9_v.csv, cas9_d.csv)
    ↓
outputfig.py / boxplot_vd.py
    ↓
箱ひげ図 + t検定結果
```

## ノーマライズ方式

`parse_intensity_text_file.py` では以下の方式でノーマライズを実行:

```
Normalized Ave = Raw Ave * (Grand Median / Image Med Average)
```

- **Grand Median**: 全データのleft_med/right_medの中央値
- **Image Med Average**: 各画像内のleft_med/right_medの平均値

## 統計検定

箱ひげ図では独立2群のt検定を実施し、有意差をアスタリスクで表示:

| 記号 | p値 |
|------|-----|
| ***** | p < 0.00001 |
| **** | p < 0.0001 |
| *** | p < 0.001 |
| ** | p < 0.01 |
| * | p < 0.05 |
| n.s. | p >= 0.05 |
