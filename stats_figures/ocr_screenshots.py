import os
import re
import csv
import json
import cv2
import pytesseract

# --- 設定（ハードコード） ---
INPUT_DIR = 'screenshots_renamed'   # PNGが入っているフォルダ
ROI_FILE = 'ocr_roi.json'           # 選択した範囲の保存先（次回以降はこれを再利用）
OUTPUT_CSV = 'ocr_result.csv'       # 抽出結果の出力先
RESELECT_ROI = False                # True にすると保存済みの範囲を無視して選び直す
CROP_DIR = 'ocr_crops'              # OCRに渡した前処理後の画像の保存先（確認用、Noneで保存しない）

# OCR設定 (Tesseract)
# --psm 6: 領域を「1つのテキストブロック」として読む（表・数値の列向き）
# 文字を数字などに限定したい場合は tessedit_char_whitelist を指定する
TESSERACT_CONFIG = '--psm 6'
# TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789.,-'
TESSERACT_LANG = 'eng'              # 日本語も読む場合は 'eng+jpn'（tesseract-ocr-jpn が必要）
UPSCALE = 3                         # スクショの小さい文字を拡大してから読む倍率
DISPLAY_MAX = 1400                  # 範囲選択ウィンドウの最大表示サイズ（px）

# 認識結果のチェック
# 領域内に出てくる決まった文字列（空白を含んでもよい）。これと数値以外が認識されたら警告を出す
ALLOWED_STRINGS = ['Mean', 'Median']
NUMBER_PATTERN = re.compile(r'^-?\d+\.\d+$')   # 小数点ありの数値（整数も許す場合は r'^-?\d+(\.\d+)?$'）


def natural_key(file_name):
    """ '2-R.png' が '10-R.png' より前に来るよう、数字部分を数値として比較する """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', file_name)]


def select_roi(image_path):
    """
    画像をポップアップ表示し、マウスで四角形の範囲を選択させる。
    戻り値は元画像サイズでの (x, y, w, h)。
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 画面に収まるよう縮小して表示し、座標は元サイズに戻す
    scale = min(1.0, DISPLAY_MAX / max(h, w))
    display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    # ウィンドウ名は日本語だとWindowsで文字化けするため英語にする
    window = 'Drag to select ROI -> Enter/Space: OK, c: cancel'
    x, y, rw, rh = cv2.selectROI(window, display, showCrosshair=True)
    cv2.destroyAllWindows()

    if rw == 0 or rh == 0:
        return None
    return [int(x / scale), int(y / scale), int(rw / scale), int(rh / scale)]


def load_or_select_roi(image_files):
    if os.path.exists(ROI_FILE) and not RESELECT_ROI:
        with open(ROI_FILE, 'r', encoding='utf-8') as f:
            roi = json.load(f)['roi']
        print(f"📐 保存済みの範囲を使用します: {roi} ({ROI_FILE})")
        return roi

    print(f"🖱️ {os.path.basename(image_files[0])} で範囲を選択してください。")
    roi = select_roi(image_files[0])
    if roi is None:
        return None

    with open(ROI_FILE, 'w', encoding='utf-8') as f:
        json.dump({'roi': roi, 'selected_on': os.path.basename(image_files[0])}, f, ensure_ascii=False, indent=2)
    print(f"✅ 範囲 {roi} を {ROI_FILE} に保存しました。")
    return roi


def preprocess(crop):
    """ グレースケール → 拡大 → 大津の2値化（背景が暗ければ反転して白背景・黒文字に揃える） """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def find_unexpected_tokens(line):
    """ 決まった文字列を取り除いたあと、小数点ありの数値に当てはまらないトークンを返す """
    # 長い文字列から先に消す（'Mean gray' と 'Mean' の両方がある場合など）
    for s in sorted(ALLOWED_STRINGS, key=len, reverse=True):
        line = line.replace(s, ' ')
    return [t for t in line.split() if not NUMBER_PATTERN.match(t)]


def ocr_image(image_path, roi):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 警告: 読み込みに失敗しました: {image_path}")
        return None

    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        print(f"⚠️ 警告: 範囲が画像の外です: {image_path} (画像サイズ {img.shape[1]}x{img.shape[0]})")
        return None

    binary = preprocess(crop)
    if CROP_DIR:
        cv2.imwrite(os.path.join(CROP_DIR, os.path.basename(image_path)), binary)

    return pytesseract.image_to_string(binary, lang=TESSERACT_LANG, config=TESSERACT_CONFIG)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"エラー: フォルダが見つかりません: {INPUT_DIR}")
        return

    image_files = [os.path.join(INPUT_DIR, f)
                   for f in sorted(os.listdir(INPUT_DIR), key=natural_key) if f.lower().endswith('.png')]
    if not image_files:
        print(f"エラー: {INPUT_DIR} にPNGファイルが見つかりません。")
        return

    roi = load_or_select_roi(image_files)
    if roi is None:
        print("範囲が選択されませんでした。終了します。")
        return

    if CROP_DIR:
        os.makedirs(CROP_DIR, exist_ok=True)

    # 1行 = 画像内のテキスト1行。空白区切りで col1, col2, ... に分割する
    # 想定外の文字が認識された行は warning 列にそのトークンを書く
    rows = []
    warning_count = 0
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        text = ocr_image(image_path, roi)
        if text is None:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line_no, line in enumerate(lines, 1):
            unexpected = find_unexpected_tokens(line)
            if unexpected:
                warning_count += 1
                print(f"⚠️ 警告: {image_name} {line_no}行目に想定外の文字があります: {unexpected}  (行: {line})")
            rows.append([image_name, line_no, line, ' '.join(unexpected)] + line.split())
        print(f"🔍 {image_name}: {len(lines)}行")

    max_cols = max((len(r) - 4 for r in rows), default=0)
    header = ['image', 'line', 'text', 'warning'] + [f'col{i}' for i in range(1, max_cols + 1)]

    # utf-8-sig: Excelで開いたときに文字化けしないようにする
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ {len(image_files)}枚から {len(rows)}行を {OUTPUT_CSV} に出力しました。")
    if warning_count:
        print(f"⚠️ 想定外の文字を含む行が {warning_count}行あります。CSVの warning 列を確認してください。")
    if CROP_DIR:
        print(f"   OCRに渡した画像は {CROP_DIR}/ で確認できます。")


# --- 実行部分 ---
main()
