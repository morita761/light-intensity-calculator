"""
ocr_screenshots.py の出力 (ocr_result-<NAME>.csv) から、V-D分岐の輝度を1測定1行にまとめたCSVを作る。

入力: スクショ1枚ごとに「平均 / 標準偏差 / 中間値 / 全ピクセル」の4行が OCR されている
      ファイル名は <測定番号>-V.png（ventral branch）と <測定番号>-D.png（dorsal branch）の組
      1行目 = 平均値, 3行目 = 中間値 を使う

出力CSVの列（変数の一覧）:
  row            : 行番号（個数・総数の確認用）。複数ファイルを指定したときは通し番号で足していく
  ventral_mean   : ventral branch の輝度平均値（<n>-V.png の1行目）
  ventral_median : ventral branch の輝度中央値（<n>-V.png の3行目）
  dorsal_mean    : dorsal branch の輝度平均値（<n>-D.png の1行目）
  dorsal_median  : dorsal branch の輝度中央値（<n>-D.png の3行目）
  image_name     : 画像名（ocr_crops-<NAME> の <NAME> の部分）
  individual_id  : 個体ID（同一個体には同じIDを付与）
  flipped        : 左右反転の有無（生理学的な Ventral-Dorsal 方向に合わせるため、ImageJ で crop したときに反転したか）
  measurement_no : 画像内の測定番号（<n>-V.png / <n>-D.png の n）。ファイルごとの番号をそのまま使う
  position       : 組織内での位置（Ventral / Middle / Dorsal の区分）。番号を書き込んだ写真の座標から決める（下記）

★ 出力後に手動で確認・編集が必要な列:
  individual_id  : ファイル（画像）ごとに 1, 2, 3, ... と仮に振っているだけ。
                   同じ個体の画像が複数あるときは、同じIDになるよう手動で直すこと。
  flipped        : 最初期の ImageJ による crop で反転したかどうかは OCR からはわからないので、すべて 'No' にしている。
                   反転した画像は手動で 'Yes' に直すこと。（0/1 だとぱっと見でわかりにくいので Yes/No にしている）
  position       : 番号を書き込んだ写真（LABEL_IMAGE_TEMPLATE）から extract_label_coords.py で番号の座標を取り、
                   番号の x 座標の左端〜右端を3等分して、左 1/3 = Ventral, 真ん中 = Middle, 右 1/3 = Dorsal にしている。
                   座標は写真ごとに label_coords-<NAME>.csv に保存し、次回以降はそれを再利用する（番号の読み間違いは
                   この CSV を手で直せばよい。写真を差し替えたときは RE_EXTRACT_LABELS = True にする）。
                   写真が無い・番号が見つからない測定は空欄。左右反転した写真では左が Ventral にならないので注意。
                   確認用画像 label_coords_check-<NAME>.png に区分の境界線を描いているので、それを見て確認すること。

その他の確認事項:
  - OCR で読めなかった値（ocr_result の warning 列に何か入っている値）や、V/D の片方しかない測定は空欄にして警告を出す。
  - 範囲のずれで行が欠けたり増えたりすると別の値を読んでしまうので、1枚の行数が EXPECTED_LINES でない画像と、
    平均が小数・中間値が整数の形になっていない値も空欄にして警告を出す。
    元のスクショを見て手入力すること。
"""
import os
import re
import csv
import cv2
from extract_label_coords import extract_label_coords, load_label_coords

# --- 設定（ハードコード） ---
# 読み込む画像セット名（ocr_screenshots.py の NAME）。複数並べると上から順に連結する
NAMES = [
    '20241024_controlFz003',
    '241106_Fz_control003',
]
INPUT_TEMPLATE = 'ocr_result-{}.csv'   # NAME から入力ファイル名を作る
OUTPUT_CSV = 'vd_intensity.csv'

MEAN_LINE = 1                          # 平均値が書かれている行（スクショ内の何行目か）
MEDIAN_LINE = 3                        # 中間値が書かれている行
EXPECTED_LINES = 4                     # 1枚あたりの行数（平均 / 標準偏差 / 中間値 / 全ピクセル）
# 値の形のチェック（行がずれて標準偏差や全ピクセルを読んでいないかの確認）
VALUE_PATTERNS = {
    MEAN_LINE: re.compile(r'^\d+\.\d+$'),    # 平均は小数
    MEDIAN_LINE: re.compile(r'^\d+$'),        # 中間値は整数
}
DEFAULT_FLIPPED = 'No'                 # 左右反転の初期値（手動で確認して Yes に直す）
POSITIONS = ['Ventral', 'Middle', 'Dorsal']  # 組織内での位置の区分（写真の左から順に割り当てる）

# 番号を書き込んだ写真（画像セットごとに1枚）と、そこから取った座標の保存先
LABEL_IMAGE_TEMPLATE = 'label-{}.png'
LABEL_COORDS_TEMPLATE = 'label_coords-{}.csv'
LABEL_CHECK_TEMPLATE = 'label_coords_check-{}.png'
RE_EXTRACT_LABELS = False              # True にすると保存済みの座標を無視して写真から取り直す

HEADER = ['row', 'ventral_mean', 'ventral_median', 'dorsal_mean', 'dorsal_median',
          'image_name', 'individual_id', 'flipped', 'measurement_no', 'position']

FILE_PATTERN = re.compile(r'^(\d+)-([VD])\.png$')   # '12-V.png' → ('12', 'V')


def load_ocr_result(path):
    """
    ocr_result の CSV を読み、{測定番号: {'V': {行番号: (値, 警告)}, 'D': {...}}} を返す
    """
    data = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            m = FILE_PATTERN.match(r['image'])
            if not m:
                print(f"⚠️ 警告: {path}: ファイル名が '<番号>-V.png' / '<番号>-D.png' の形ではないため無視します: {r['image']}")
                continue
            no, branch = int(m.group(1)), m.group(2)
            data.setdefault(no, {}).setdefault(branch, {})[int(r['line'])] = (r['text'], r['warning'])
    return data


def get_value(data, name, no, branch, line_no):
    """ 値を OCR で読んだ文字列のまま返す（中央値は整数のまま）。無い・OCR で読めていない場合は警告を出して空欄にする """
    lines = data[no].get(branch)
    if lines is None:
        print(f"⚠️ 警告: {name} 測定番号 {no}: {no}-{branch}.png がありません（空欄にします）")
        return ''
    if line_no not in lines:
        print(f"⚠️ 警告: {name} {no}-{branch}.png に {line_no}行目がありません（空欄にします）")
        return ''
    if len(lines) != EXPECTED_LINES:
        print(f"⚠️ 警告: {name} {no}-{branch}.png の行数が {len(lines)}行です（{EXPECTED_LINES}行のはず）。"
              f"範囲がずれている可能性があるため {line_no}行目を空欄にします")
        return ''
    text, warning = lines[line_no]
    if warning:
        print(f"⚠️ 警告: {name} {no}-{branch}.png の {line_no}行目が OCR で読めていません: {text}（空欄にします）")
        return ''
    pattern = VALUE_PATTERNS.get(line_no)
    if pattern and not pattern.match(text):
        print(f"⚠️ 警告: {name} {no}-{branch}.png の {line_no}行目の形が想定と違います: {text}（空欄にします）")
        return ''
    return text


def load_positions(name):
    """
    写真の番号の座標から {測定番号: 区分} を返す。座標が取れない場合は空の dict を返す。
    x 座標の左端〜右端を len(POSITIONS) 等分し、左から POSITIONS を割り当てる。
    """
    coords_path = LABEL_COORDS_TEMPLATE.format(name)
    extracted = RE_EXTRACT_LABELS or not os.path.exists(coords_path)
    if extracted or not os.path.exists(coords_path):
        image_path = LABEL_IMAGE_TEMPLATE.format(name)
        if not os.path.exists(image_path):
            print(f"⚠️ 警告: {name}: 番号を書き込んだ写真 {image_path} がありません（position は空欄にします）")
            return {}
        print(f"🔍 {name}: {image_path} から番号の座標を取ります")
        if not extract_label_coords(image_path, coords_path, LABEL_CHECK_TEMPLATE.format(name)):
            return {}
    else:
        print(f"📐 {name}: 保存済みの座標を使用します ({coords_path})")

    coords = load_label_coords(coords_path)
    if len(coords) < 2:
        print(f"⚠️ 警告: {name}: 番号が {len(coords)}個しか見つからないため区分を決められません（position は空欄にします）")
        return {}

    xs = [x for x, _ in coords.values()]
    x_min, x_max = min(xs), max(xs)
    width = (x_max - x_min) / len(POSITIONS)
    bounds = [x_min + width * i for i in range(1, len(POSITIONS))]   # 区分の境界の x 座標

    # 写真から取り直したときだけ、確認用画像に区分の境界線を描き足す（再実行のたびに重ね描きしないため）
    check_path = LABEL_CHECK_TEMPLATE.format(name)
    check = cv2.imread(check_path) if extracted else None
    if check is not None:
        for b in bounds:
            cv2.line(check, (int(b), 0), (int(b), check.shape[0]), (255, 255, 255), 2)
        for i, p in enumerate(POSITIONS):
            cv2.putText(check, p, (int(x_min + width * i) + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(check_path, check)

    return {no: POSITIONS[min(int((x - x_min) / width), len(POSITIONS) - 1)] if width > 0 else POSITIONS[0]
            for no, (x, _) in coords.items()}


def main():
    rows = []
    for individual_id, name in enumerate(NAMES, 1):
        path = INPUT_TEMPLATE.format(name)
        if not os.path.exists(path):
            print(f"エラー: ファイルが見つかりません: {path}")
            return

        data = load_ocr_result(path)
        numbers = sorted(data)
        positions = load_positions(name)
        if positions:
            no_label = [no for no in numbers if no not in positions]
            if no_label:
                print(f"⚠️ 警告: {name}: 写真に番号が見つからない測定があります（position は空欄にします）: {no_label}")
            extra = sorted(set(positions) - set(numbers))
            if extra:
                print(f"⚠️ 警告: {name}: 写真にあるが測定データが無い番号があります: {extra}")
        for no in numbers:
            rows.append([
                len(rows) + 1,
                get_value(data, name, no, 'V', MEAN_LINE),
                get_value(data, name, no, 'V', MEDIAN_LINE),
                get_value(data, name, no, 'D', MEAN_LINE),
                get_value(data, name, no, 'D', MEDIAN_LINE),
                name,
                individual_id,
                DEFAULT_FLIPPED,
                no,
                positions.get(no, ''),
            ])

        counts = [sum(1 for no in numbers if positions.get(no) == p) for p in POSITIONS]
        print(f"📄 {name}: {len(numbers)}測定（個体ID {individual_id}, "
              + ' / '.join(f'{p} {c}' for p, c in zip(POSITIONS, counts)) + '）')

    # utf-8-sig: Excelで開いたときに文字化けしないようにする
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"\n✅ {len(NAMES)}ファイルから {len(rows)}行を {OUTPUT_CSV} に出力しました。")
    print("   individual_id / flipped / position は仮の値です。手動で確認してください（詳細はファイル冒頭のコメント）。")


# --- 実行部分 ---
main()
