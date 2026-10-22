"""
画像に白文字で書き込まれた番号（馬蹄形の測定番号など）を見つけ、番号ごとの座標をCSVに出力する。

処理の流れ:
  1. R, G, B すべてが WHITE_THRESHOLD より明るい画素 = 白文字 として取り出す
     （緑の蛍光・水色/青/マゼンタの馬蹄形の線は、どれかのチャンネルが暗いので除外される）
  2. 白い塊のうち、高さ・幅が数字1文字らしいものだけを残す
  3. 横に並んでいる数字（'1' と '0' など）をまとめて1つの番号にする
  4. 番号ごとに Tesseract で読み取る
  5. 番号の中心座標と外接矩形をCSVに出力し、確認用に枠と読み取り結果を描いた画像も保存する

座標は入力画像のピクセル座標（左上が原点, x は右向き, y は下向き）。
スクリーンショットを入力にした場合はスクショ上の座標なので、元の顕微鏡画像の座標とは拡大率・位置がずれる。
"""
import csv
from collections import Counter
import cv2
import numpy as np
import pytesseract

# --- 設定（ハードコード） ---
INPUT_IMAGE = 'label_image.png'          # 番号が書き込まれた画像
OUTPUT_CSV = 'label_coords.csv'          # 番号ごとの座標の出力先
OUTPUT_IMAGE = 'label_coords_check.png'  # 確認用画像（検出した枠と読み取った番号を描画）
ROI = None                               # 探す範囲 [x, y, w, h]。None なら画像全体（ステータスバーなどを除きたいときに指定）

WHITE_THRESHOLD = 150                    # R, G, B すべてがこの値より明るい画素を白文字とみなす
DIGIT_HEIGHT = (20, 50)                  # 数字1文字の高さ（px）の範囲。画像の解像度が変わったら合わせる
GROUP_Y_TOL = 0.3                        # 同じ番号とみなす数字の上端のずれ（文字の高さに対する割合）
GROUP_X_GAP = 0.6                        # 同じ番号とみなす数字の間隔の上限（文字の高さに対する割合）
OCR_PAD = 20                             # Tesseract に渡すときの余白（px）。狭いと1文字の数字が読めないことがある
OCR_SCALE = 3                            # Tesseract に渡すときの拡大倍率
# Tesseract は読み取りモード（psm）によって読めない数字が変わる（psm 7 は 1文字の '1' '4' を、psm 8 は '5' '11' を落とした）
# そのため複数のモードで読み、いちばん多かった結果を採用する（数字だけに限定）
TESSERACT_PSMS = [7, 8, 10, 13]   # 7: 1行, 8: 1単語, 10: 1文字, 13: 生の1行
TESSERACT_WHITELIST = '0123456789'


def find_digits(mask):
    """ 数字1文字らしい白い塊を [(x, y, w, h, ラベル番号), ...] で返す """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    digits = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        # 高さが範囲内で、横長でないもの（ホームバーなどの横線を除く）
        if DIGIT_HEIGHT[0] <= h <= DIGIT_HEIGHT[1] and w < h:
            digits.append((x, y, w, h, i))
    return digits, labels


def group_digits(digits):
    """ 左から順に見て、直前の数字のすぐ右・同じ高さにある数字を同じ番号にまとめる """
    groups = []
    for d in sorted(digits, key=lambda d: d[0]):
        x, y, w, h, _ = d
        for g in groups:
            gx, gy, gw, gh, _ = g[-1]
            if abs(y - gy) < gh * GROUP_Y_TOL and 0 <= x - (gx + gw) < gh * GROUP_X_GAP:
                g.append(d)
                break
        else:
            groups.append([d])
    return groups


def read_number(group, labels):
    """ 番号を構成する塊だけを白背景・黒文字で描き直し、Tesseract で読む """
    x0 = min(d[0] for d in group)
    y0 = min(d[1] for d in group)
    x1 = max(d[0] + d[2] for d in group)
    y1 = max(d[1] + d[3] for d in group)

    # 近くの別の番号が入らないよう、この番号の塊だけを描く
    region = labels[y0:y1, x0:x1]
    glyph = np.isin(region, [d[4] for d in group])
    canvas = np.full((y1 - y0 + 2 * OCR_PAD, x1 - x0 + 2 * OCR_PAD), 255, np.uint8)
    canvas[OCR_PAD:OCR_PAD + (y1 - y0), OCR_PAD:OCR_PAD + (x1 - x0)][glyph] = 0
    canvas = cv2.resize(canvas, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)

    reads = [pytesseract.image_to_string(canvas, config=f'--psm {psm} -c tessedit_char_whitelist={TESSERACT_WHITELIST}').strip()
             for psm in TESSERACT_PSMS]
    votes = Counter(r for r in reads if r)
    text = votes.most_common(1)[0][0] if votes else ''
    # モード間で読み取りが割れたものは誤認識の可能性があるので知らせる
    if len(votes) > 1:
        print(f"⚠️ 警告: 中心 ({(x0 + x1) / 2:.0f}, {(y0 + y1) / 2:.0f}) の読み取りがモードで割れました: {reads} → '{text}' を採用")
    return text, (x0, y0, x1 - x0, y1 - y0)


def main():
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"エラー: 画像が読み込めません: {INPUT_IMAGE}")
        return

    # ROI の外は探さない（座標は画像全体の座標のまま扱う）
    mask = (img.min(axis=2) > WHITE_THRESHOLD).astype(np.uint8)
    if ROI is not None:
        rx, ry, rw, rh = ROI
        roi_mask = np.zeros_like(mask)
        roi_mask[ry:ry + rh, rx:rx + rw] = 1
        mask *= roi_mask

    digits, labels = find_digits(mask)
    groups = group_digits(digits)
    print(f"🔍 数字 {len(digits)}文字 → 番号 {len(groups)}個 を検出しました。")

    results = []
    for group in groups:
        text, (x, y, w, h) = read_number(group, labels)
        results.append({'text': text, 'x': x + w / 2, 'y': y + h / 2, 'bbox': (x, y, w, h)})

    # --- 読み取り結果のチェック ---
    numbers = []
    for r in results:
        if not r['text'].isdigit():
            print(f"⚠️ 警告: 中心 ({r['x']:.0f}, {r['y']:.0f}) の番号が読めませんでした: '{r['text']}'")
        else:
            numbers.append(int(r['text']))
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        print(f"⚠️ 警告: 同じ番号が複数あります: {duplicates}")
    if numbers:
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        if missing:
            print(f"⚠️ 警告: 1〜{max(numbers)} のうち見つからない番号があります: {missing}")

    # 番号順（読めなかったものは最後）に並べる
    results.sort(key=lambda r: (not r['text'].isdigit(), int(r['text']) if r['text'].isdigit() else 0))

    # utf-8-sig: Excelで開いたときに文字化けしないようにする
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['number', 'x', 'y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        for r in results:
            writer.writerow([r['text'], f"{r['x']:.1f}", f"{r['y']:.1f}", *r['bbox']])
    print(f"✅ {len(results)}個の番号の座標を {OUTPUT_CSV} に出力しました。")

    # 確認用画像: 検出枠（黄）と読み取った番号（読めなかったものは赤で '?'）
    check = img.copy()
    for r in results:
        x, y, w, h = r['bbox']
        ok = r['text'].isdigit()
        color = (0, 255, 255) if ok else (0, 0, 255)
        cv2.rectangle(check, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 2)
        cv2.putText(check, r['text'] if ok else '?', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(OUTPUT_IMAGE, check)
    print(f"   確認用画像を {OUTPUT_IMAGE} に保存しました。")


# --- 実行部分 ---
main()
