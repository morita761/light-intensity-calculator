import os
import re
import shutil
from datetime import datetime

# --- 設定（ハードコード） ---
INPUT_DIR = 'screenshots'           # スクリーンショットが入っているフォルダ
OUTPUT_DIR = 'screenshots_renamed'  # リネーム後のコピー先フォルダ（元ファイルは変更しない）
START_NUMBER = 1                    # 開始番号（例: 5 にすると 5-R, 5-L, 6-R, ... ）
SIDES = ['R', 'L']                  # 2枚一組の並び順（1枚目がR、2枚目がL）

# ファイル名中の日時パターン
# 例: "スクリーンショット 2026-10-21 9.05.03.png" (macOS), "Screenshot 2026-10-21 090503.png"
date_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})\D+(\d{1,2})[.:]?(\d{2})[.:]?(\d{2})')


def get_sort_key(file_path):
    """
    ファイル名から撮影日時を取得する。取得できない場合は更新日時を使う。
    (macOSのスクショ名は時が0埋めされないため、名前順ソートだと 10時 < 9時 になる)
    """
    match = date_pattern.search(os.path.basename(file_path))
    if match:
        return datetime(*map(int, match.groups()))
    return datetime.fromtimestamp(os.path.getmtime(file_path))


def rename_screenshots(input_dir, output_dir, start_number):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    if not files:
        print(f"エラー: {input_dir} にPNGファイルが見つかりません。")
        return

    files.sort(key=get_sort_key)

    if len(files) % 2 != 0:
        print(f"⚠️ 警告: ファイル数が奇数 ({len(files)}枚) です。最後の1枚はペアになりません。")

    os.makedirs(output_dir, exist_ok=True)

    for i, src in enumerate(files):
        number = start_number + i // len(SIDES)
        side = SIDES[i % len(SIDES)]
        dst = os.path.join(output_dir, f"{number}-{side}.png")
        shutil.copy2(src, dst)
        print(f"{os.path.basename(src)} -> {os.path.basename(dst)}")

    print(f"\n✅ {len(files)}枚を {output_dir} に出力しました。")


# --- 実行部分 ---
rename_screenshots(INPUT_DIR, OUTPUT_DIR, START_NUMBER)
