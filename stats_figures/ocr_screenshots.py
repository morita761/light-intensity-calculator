import os
import re
import csv
import json
import cv2
import numpy as np

# --- 設定（ハードコード） ---
NAME = '20241024_controlFz003'
INPUT_DIR = 'rename-'+NAME   # PNGが入っているフォルダ
ROI_FILE = 'ocr_roi.json'           # 選択した範囲の保存先（次回以降はこれを再利用）
OUTPUT_CSV = 'ocr_result-'+NAME+'.csv'  # 抽出結果の出力先（ocr_to_vd_csv.py がこの名前で読み込む）
RESELECT_ROI = True                # True にすると保存済みの範囲を無視して選び直す
CROP_DIR = 'ocr_crops-'+NAME             # OCRに渡した前処理後の画像の保存先（確認用、Noneで保存しない）

# 文字認識設定（テンプレート照合）
# 画面の文字はスクショごとにピクセル単位で同じ形なので、Tesseractではなく文字の形をテンプレートと完全一致で照合する
# （Tesseractはこのドットフォントの 5 を $ や 3 と誤認識するため使わない）
# テンプレートにない形は '?' になり警告が出る。画面の拡大率やフォントが変わったらテンプレートを作り直す必要がある
TEMPLATE_FILE = 'ocr_templates.json'  # 文字ごとの2値パターン（'#'=文字, '.'=背景）
TEXT_THRESHOLD = 150                # この輝度より明るい画素を文字とみなす（背景 約80, 文字 最大 約210）
CROP_SCALE = 8                      # 確認用画像の拡大倍率
DISPLAY_MAX = 1400                  # 範囲選択ウィンドウの最大表示サイズ（px）

# 認識結果のチェック
# 整数・小数以外（文字列）が認識されたら警告を出す
# 警告から除外したい決まった文字列があればここに追加する（例: ['Mean', 'Median']）
ALLOWED_STRINGS = []
NUMBER_PATTERN = re.compile(r'^-?\d+(\.\d+)?$')   # 整数または小数


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


def load_templates():
    """ {文字: 2値配列} を返す。幅の広いものから照合するよう並べる """
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    templates = {ch: np.array([[c == '#' for c in row] for row in rows]) for ch, rows in raw.items()}
    return sorted(templates.items(), key=lambda t: t[1].shape[1], reverse=True)


def find_runs(mask):
    """ True が連続する区間 [(開始, 終了), ...] を返す """
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def read_line(line, templates):
    """
    1行分の2値画像を左から順にテンプレートと照合して文字列にする。
    文字同士がくっついていても、テンプレートの幅ぶん完全一致すれば読める。
    どのテンプレートにも一致しない部分は '?' にして次の空白列まで飛ばす。
    """
    text = ''
    x, width = 0, line.shape[1]
    while x < width:
        if not line[:, x].any():
            x += 1
            continue
        for ch, tpl in templates:
            tw = tpl.shape[1]
            if tpl.shape[0] == line.shape[0] and x + tw <= width and np.array_equal(line[:, x:x + tw], tpl):
                text += ch
                x += tw
                break
        else:
            text += '?'
            while x < width and line[:, x].any():
                x += 1
    return text


def find_unexpected_tokens(line):
    """ 決まった文字列を取り除いたあと、整数・小数に当てはまらないトークンを返す """
    # 長い文字列から先に消す（'Mean gray' と 'Mean' の両方がある場合など）
    for s in sorted(ALLOWED_STRINGS, key=len, reverse=True):
        line = line.replace(s, ' ')
    return [t for t in line.split() if not NUMBER_PATTERN.match(t)]


def ocr_image(image_path, roi, templates):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 警告: 読み込みに失敗しました: {image_path}")
        return None

    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        print(f"⚠️ 警告: 範囲が画像の外です: {image_path} (画像サイズ {img.shape[1]}x{img.shape[0]})")
        return None

    binary = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > TEXT_THRESHOLD
    if CROP_DIR:
        view = (~binary).astype(np.uint8) * 255   # 白背景・黒文字で保存
        view = cv2.resize(view, None, fx=CROP_SCALE, fy=CROP_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(CROP_DIR, os.path.basename(image_path)), view)

    # 文字のある行のまとまり = テキスト1行
    lines = [read_line(binary[r0:r1], templates) for r0, r1 in find_runs(binary.any(axis=1))]
    return '\n'.join(lines)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"エラー: フォルダが見つかりません: {INPUT_DIR}")
        return

    image_files = [os.path.join(INPUT_DIR, f)
                   for f in sorted(os.listdir(INPUT_DIR), key=natural_key) if f.lower().endswith('.png')]
    if not image_files:
        print(f"エラー: {INPUT_DIR} にPNGファイルが見つかりません。")
        return

    if not os.path.exists(TEMPLATE_FILE):
        print(f"エラー: テンプレートファイルが見つかりません: {TEMPLATE_FILE}")
        return
    templates = load_templates()

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
        text = ocr_image(image_path, roi, templates)
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
