# screenshot_ocr

Photoshop のヒストグラムパネル（平均 / 標準偏差 / 中間値 / 全ピクセル）のスクリーンショットから数値を読み取り、V-D分岐の輝度CSVを作るフォルダ。

## 手順

各スクリプト冒頭の `NAME` を画像セット名に合わせてから、このフォルダで順に実行する。

| 順 | ファイル | 説明 |
|---|---------|------|
| 1 | `rename_screenshots.py` | スクリーンショットを撮影日時順に2枚一組で `1-V`, `1-D`, `2-V`, ... にリネームし、`rename-<NAME>/` にコピー |
| 2 | `ocr_screenshots.py` | 1枚目で範囲を選択（`ocr_roi-<NAME>.json` に保存して全画像に適用）し、テンプレート照合で数値を読み取って `ocr_result-<NAME>.csv` に出力 |
| 3 | `ocr_to_vd_csv.py` | `NAMES` に並べた画像セットの OCR 結果を、1測定1行の `vd_intensity.csv` にまとめる |
| - | `compare_ocr_vs_re.py` | OCR の平均値と Photoshop の計測ログ（`<NAME>_grayvalue_RE.txt`）を比較プロット |

## 補足

- `ocr_templates.json` に数字の2値パターンを持つ。Photoshop の文字サイズ変更前（`small_8px`）と変更後（`large_9px`）の両方に対応し、行の高さで自動的に使い分ける。
- テンプレートにない形は `?` になり警告が出る（Tesseract はこのフォントの 5 を `$` や 3 と誤認識するため使っていない）。
- `vd_intensity.csv` の `individual_id` / `flipped` / `position` は仮の値なので手動で確認する（詳細は `ocr_to_vd_csv.py` 冒頭）。
