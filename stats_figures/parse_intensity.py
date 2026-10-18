"""輝度測定テキストを画像ごとにノーマライズし、系統・V/D別のCSVに分割する

用途: 全ての作図スクリプトの前処理。生データ（Detectron2等の輝度測定結果）を
      画像間の明るさの違いを補正（ノーマライズ）してCSVにする。
      生データを追加・変更したら最初にこれを実行する。
      ノーマライズ式: Normalized Ave = Raw Ave * (Grand Median / Image Med Average)
入力: data/raw/*.txt（どのファイルを使うかは config/datasets.txt で指定）
出力: data/processed/<dataset>/<ファイル名>_v.csv, _d.csv

Usage:
    python parse_intensity.py                       # datasets.txt の全セクション
    python parse_intensity.py --dataset two_groups  # 指定セクションのみ
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from lib.data_io import SIDES, normalize_by_image, parse_intensity_txt, processed_csv_path, raw_path, write_lr_csv
from lib.settings import dataset_names, load_dataset


def process_dataset(dataset):
    print("=" * 60)
    print(f"[{dataset}]")
    print("=" * 60)

    groups = load_dataset(dataset)
    stems = []
    dfs = []
    for raw in groups.values():
        path = raw_path(raw)
        if not path.exists():
            sys.exit(f"エラー: ファイルが見つかりません: {path}")
        stem = Path(raw).stem
        if stem in stems:
            continue
        stems.append(stem)
        dfs.append(parse_intensity_txt(path, stem))

    df = normalize_by_image(pd.concat(dfs, ignore_index=True))
    print("\nNormalized Ave = Raw Ave * (Grand Median / Image Med Average) で計算しました。")

    print("\n--- CSVファイルの出力 ---")
    for stem in stems:
        for sec in SIDES:
            df_out = df[(df['experiment'] == stem) & (df['section'] == sec)]
            if df_out.empty:
                print(f"スキップ: {stem} {sec}（データなし）")
                continue
            write_lr_csv(processed_csv_path(dataset, stem, sec),
                         df_out['normalized_left'], df_out['normalized_right'])
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='*', help='config/datasets.txt のセクション名（省略時は全て）')
    args = parser.parse_args()

    for dataset in args.dataset or dataset_names():
        process_dataset(dataset)
    print("--- 全ての処理が完了しました ---")


if __name__ == '__main__':
    main()
