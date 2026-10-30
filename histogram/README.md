# histogram

画像の明度をヒストグラムにして、CLAHE（`cv2.createCLAHE` による局所適応ヒストグラム平坦化）の
効果を可視化する実験フォルダ。参考: https://www.codevace.com/py-opencv-clahe/

## 使い方

```bash
python clahe_histogram.py                                   # data/up_Fz_green_stronger.tif
python clahe_histogram.py --image other.tif --clip-limit 3.0 --tile 16
python clahe_histogram.py --both --no-show                  # 通常版 + 文字無し版を保存
python clahe_histogram.py -h                                # オプション一覧
```

| パス | 内容 |
|---|---|
| `clahe_histogram.py` | 元画像/CLAHE後の画像と明度ヒストグラムを2行2列で表示・保存 |
| `config/figure_settings.txt` | 既定の入力画像・CLAHEパラメータ・タイトル・軸ラベル |
| `data/` | 入力画像 |
| `output/clahe_histogram/` | 出力（git管理外） |

`--no-text` / `--both` / `--no-show` は stats_figures と共通（`shared/figure.py`）。
