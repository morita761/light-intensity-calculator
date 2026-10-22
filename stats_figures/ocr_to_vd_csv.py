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
  position       : 組織内での位置（Ventral / Middle / Dorsal の区分）

★ 出力後に手動で確認・編集が必要な列:
  individual_id  : ファイル（画像）ごとに 1, 2, 3, ... と仮に振っているだけ。
                   同じ個体の画像が複数あるときは、同じIDになるよう手動で直すこと。
  flipped        : 最初期の ImageJ による crop で反転したかどうかは OCR からはわからないので、すべて 'No' にしている。
                   反転した画像は手動で 'Yes' に直すこと。（0/1 だとぱっと見でわかりにくいので Yes/No にしている）
  position       : ファイルごとに測定番号順で、総数の 1/3 ずつ Ventral → Middle → Dorsal と機械的に割り当てている。
                   （割り切れないときは、後ろの区分ほど少なくなる。例: 41個 → 14 / 14 / 13）
                   実際の組織上の位置と合っているか、画像を見て手動で確認すること。

その他の確認事項:
  - OCR で読めなかった値（ocr_result の warning 列に何か入っている値）や、V/D の片方しかない測定は空欄にして警告を出す。
    元のスクショを見て手入力すること。
"""
import os
import re
import csv

# --- 設定（ハードコード） ---
# 読み込む画像セット名（ocr_screenshots.py の NAME）。複数並べると上から順に連結する
NAMES = [
    '20241024_controlFz003',
]
INPUT_TEMPLATE = 'ocr_result-{}.csv'   # NAME から入力ファイル名を作る
OUTPUT_CSV = 'vd_intensity.csv'

MEAN_LINE = 1                          # 平均値が書かれている行（スクショ内の何行目か）
MEDIAN_LINE = 3                        # 中間値が書かれている行
DEFAULT_FLIPPED = 'No'                 # 左右反転の初期値（手動で確認して Yes に直す）
POSITIONS = ['Ventral', 'Middle', 'Dorsal']  # 組織内での位置の区分（測定番号の若い方から順に割り当てる）

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
    text, warning = lines[line_no]
    if warning:
        print(f"⚠️ 警告: {name} {no}-{branch}.png の {line_no}行目が OCR で読めていません: {text}（空欄にします）")
        return ''
    return text


def assign_position(index, total):
    """ 0始まりの index を、総数 total の 1/3 ずつ POSITIONS に割り当てる """
    return POSITIONS[index * len(POSITIONS) // total]


def main():
    rows = []
    for individual_id, name in enumerate(NAMES, 1):
        path = INPUT_TEMPLATE.format(name)
        if not os.path.exists(path):
            print(f"エラー: ファイルが見つかりません: {path}")
            return

        data = load_ocr_result(path)
        numbers = sorted(data)
        for i, no in enumerate(numbers):
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
                assign_position(i, len(numbers)),
            ])

        counts = [sum(1 for i in range(len(numbers)) if assign_position(i, len(numbers)) == p) for p in POSITIONS]
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
