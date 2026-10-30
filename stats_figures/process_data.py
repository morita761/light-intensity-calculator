"""1つの輝度測定テキストをV/Dに分け、旧方式でノーマライズしてCSV出力する

用途: boxplot_vd.py 用の前処理（1ファイルだけを扱う旧方式）。
      ノーマライズ式:
        Normalized Ave = Ave / (画像内の left_med平均 + right_med平均)
                             * (全体の left_med平均 + right_med平均)
入力: data/raw/<input>（既定: control.txt）
出力: data/processed/single/v.csv, d.csv（元の値）, v_normalize.csv, d_normalize.csv

Usage:
    python process_data.py
    python process_data.py --input vang_rnai.txt
"""
import argparse
import sys

from lib.data_io import parse_intensity_txt, raw_path
from lib.paths import PROCESSED_DIR

OUTPUT_DIR = PROCESSED_DIR / 'single'


def write_csv(filename, rows, fmt):
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write("left_intensity right_intensity\n")
        for left, right in rows:
            f.write(f"{left:{fmt}} {right:{fmt}}\n")
    print(f"{len(rows)}行を {path.relative_to(PROCESSED_DIR.parent.parent)} に出力しました。")


def process_text_file(input_file):
    path = raw_path(input_file)
    if not path.exists():
        sys.exit(f"エラー: ファイルが見つかりません: {path}")
    df = parse_intensity_txt(path, path.stem)
    if df.empty:
        sys.exit("エラー: データがありません。")

    # 全体の中央値の平均の合計（ノーマライズ定数）
    total_avg_left_med = df['left_med'].mean()
    total_avg_right_med = df['right_med'].mean()
    normalization_constant = total_avg_left_med + total_avg_right_med
    if normalization_constant == 0:
        sys.exit("エラー: 全体の中央値の平均の合計が0です。ノーマライズを実行できません。")

    # 画像（区分）ごとの中央値の平均の合計
    by_title = df.groupby('title', sort=False)[['left_med', 'right_med']].mean()
    group_med_sums = by_title['left_med'] + by_title['right_med']

    raw_rows = {'V': [], 'D': []}
    norm_rows = {'V': [], 'D': []}
    for row in df.itertuples():
        group_med_sum = group_med_sums[row.title]
        if group_med_sum == 0:
            print(f"警告: 区分 {row.title} の中央値の合計が0です。スキップします。")
            continue
        raw_rows[row.section].append((row.left_ave, row.right_ave))
        norm_rows[row.section].append((row.left_ave / group_med_sum * normalization_constant,
                                       row.right_ave / group_med_sum * normalization_constant))

    write_csv('v_normalize.csv', norm_rows['V'], '')
    write_csv('d_normalize.csv', norm_rows['D'], '')
    write_csv('v.csv', raw_rows['V'], '.3f')
    write_csv('d.csv', raw_rows['D'], '.3f')

    print("\n--- 統計情報：中央値 (Med) の平均 ---")
    print(f"【全体平均に使用される定数】: {normalization_constant:.2f} "
          f"({total_avg_left_med:.2f} + {total_avg_right_med:.2f})")
    for title, row in by_title.iterrows():
        print(f"【区分: {title}】")
        print(f"  > left_med の平均: {row['left_med']:.2f}")
        print(f"  > right_med の平均: {row['right_med']:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='control.txt', help='data/raw/ 内のファイル名 (既定: control.txt)')
    process_text_file(parser.parse_args().input)
