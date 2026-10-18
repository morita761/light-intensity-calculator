# horseShoeShapeRecog

深層学習を使わず、OpenCV だけで馬蹄形を抽出しようとした初期の実験（2025年5月ごろ）。
現在のメインパイプラインは `Detectron2/`（検出）→ `stats_figures/` / `heatmap/`（解析）。

ルートの `main.py` / `horseshoe.py` から `import horseShoeShapeRecog.xxx` で使われているため、
フォルダ名・ファイル名は変えないこと。

| ファイル | 内容 |
|---|---|
| `Dog.py` | グレースケール → CLAHE → 適応的閾値 → クロージング → 輪郭抽出で馬蹄形の候補を描く（`extract_horseshoe_shape`） |
| `ellipseAppro.py` | 輪郭が馬蹄形らしいかを面積と楕円近似の誤差で判定（`is_horseshoe_like`、認識は未確認） |
| `gptFunc.py` | 緑色領域のマスク（HSV）と、最大輪郭を馬蹄形とみなす抽出 |

単体で実行するスクリプトではない（`python main.py` などから呼ばれる）。
