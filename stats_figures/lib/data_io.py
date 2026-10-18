"""輝度データの読み込み・ノーマライズ・CSV入出力"""
import re
import sys
from pathlib import Path

import pandas as pd

from .paths import RAW_DIR, PROCESSED_DIR
from .settings import load_dataset

# 解析パターン: (Index), (Left Ave), (Left Med), (Right Ave), (Right Med)
DATA_PATTERN = re.compile(r'^\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,')

SIDES = {'V': 'Ventral', 'D': 'Dorsal'}
COLUMNS = ['left_intensity', 'right_intensity']


def parse_intensity_txt(input_file_path, experiment_name):
    """輝度測定結果のテキストを読み込み DataFrame にする

    テキストの書式:
        - <画像タイトル>
        V
        1, <left_ave>, <left_med>, <right_ave>, <right_med>,
        ...
        D
        ...

    返り値の列: experiment, section(V/D), title, left_ave, right_ave, left_med, right_med
    """
    input_file_path = Path(input_file_path)
    all_data = []
    current_section = None
    current_title = None

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, original_line in enumerate(f, 1):
            line = original_line.strip()
            if not line:
                continue

            if line.startswith('-'):
                current_title = line.lstrip('-').strip()  # 画像ID/タイトル
                current_section = None
                continue

            if line in SIDES:
                current_section = line
                continue

            if not (current_section and current_title):
                continue

            match = DATA_PATTERN.match(line)
            if match:
                _, left_ave, left_med, right_ave, right_med = match.groups()
                try:
                    all_data.append([
                        experiment_name, current_section, current_title,
                        float(left_ave), float(right_ave),
                        int(left_med), int(right_med),
                    ])
                except ValueError as e:
                    print(f"警告: {input_file_path.name} 行{line_num}の数値変換に失敗しました: {line} (エラー: {e})")
            elif '/' in line:
                print(f"警告: {input_file_path.name} 行{line_num}で特殊なデータ形式が見つかりました（スキップ）: {line}")

    df = pd.DataFrame(all_data, columns=['experiment', 'section', 'title',
                                         'left_ave', 'right_ave', 'left_med', 'right_med'])
    print(f"{input_file_path.name} から {len(df)} 行のデータを読み込みました。")
    return df


def normalize_by_image(df_combined):
    """画像ごとにノーマライズする

    Normalized Ave = Raw Ave * (Grand Median / Image Med Average)
      - Grand Median     : 全データの left_med / right_med の中央値
      - Image Med Average: 各画像内（V/D両方）の left_med / right_med の平均値
    """
    df = df_combined.copy()

    grand_median = pd.concat([df['left_med'], df['right_med']]).dropna().median()
    if grand_median == 0:
        raise ValueError("全データの中央値が0のため、ノーマライズできません。")
    print(f"\nグローバルノーマライズ定数 (Grand Median): {grand_median:.4f}")

    df['image_key'] = df['experiment'] + '_' + df['title']
    print("\n画像ごとのローカルノーマライズファクター (Image Med Average):")
    image_med_avgs = {}
    for key, df_image in df.groupby('image_key', sort=False):
        value = pd.concat([df_image['left_med'], df_image['right_med']]).dropna().mean()
        image_med_avgs[key] = value
        print(f"   - {key}: {value:.4f}")
    df['image_med_avg'] = df['image_key'].map(image_med_avgs)

    if (df['image_med_avg'] == 0).any():
        raise ValueError("いずれかの画像のグループ中央値平均が0です。ノーマライズできません。")

    df['scaling_factor'] = grand_median / df['image_med_avg']
    df['normalized_left'] = df['left_ave'] * df['scaling_factor']
    df['normalized_right'] = df['right_ave'] * df['scaling_factor']
    return df


def processed_csv_path(dataset, stem, section):
    """data/processed/<dataset>/<stem>_<v|d>.csv"""
    return PROCESSED_DIR / dataset / f'{stem}_{section.lower()}.csv'


def write_lr_csv(path, left, right, float_format='%.6f'):
    """left_intensity / right_intensity の2列をスペース区切りCSVで保存"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({COLUMNS[0]: list(left), COLUMNS[1]: list(right)})
    df.to_csv(path, sep=' ', index=False, float_format=float_format, lineterminator='\n')
    print(f"{len(df)}行を {path.relative_to(PROCESSED_DIR.parent.parent)} に出力しました。")


def read_lr_csv(path):
    """スペース区切りの left_intensity / right_intensity CSV を読み込む"""
    path = Path(path)
    if not path.exists():
        sys.exit(f"エラー: ファイルが見つかりません: {path}\n"
                 f"先に `python parse_intensity.py` を実行してCSVを生成してください。")
    return pd.read_csv(path, sep=r'\s+')


def load_vd(dataset):
    """datasets.txt の系統ごとに V/D のCSVを読み込む

    返り値: {表示名: {'V': DataFrame, 'D': DataFrame}}（datasets.txt の記載順）
    """
    groups = load_dataset(dataset)
    return {
        label: {sec: read_lr_csv(processed_csv_path(dataset, Path(raw).stem, sec)) for sec in SIDES}
        for label, raw in groups.items()
    }


def raw_path(filename):
    return RAW_DIR / filename


def abs_diff_long(data, side_names=SIDES):
    """|left - right| を縦持ち DataFrame (abs_diff, genotype, side) にする"""
    frames = []
    for genotype, by_side in data.items():
        for sec, df in by_side.items():
            diff = (df[COLUMNS[0]] - df[COLUMNS[1]]).abs().to_frame(name='abs_diff')
            diff['genotype'] = genotype
            diff['side'] = side_names[sec]
            frames.append(diff)
    return pd.concat(frames, ignore_index=True)


def intensity_long(data, branch_labels=None):
    """左右の輝度を縦持ち DataFrame (intensity, branch, genotype, side) にする

    branch_labels: (left の表示名, right の表示名)。None なら列名のまま。
    """
    rename = dict(zip(COLUMNS, branch_labels)) if branch_labels else {}
    frames = []
    for genotype, by_side in data.items():
        for sec, df in by_side.items():
            melted = df[COLUMNS].melt(var_name='branch', value_name='intensity')
            melted['branch'] = melted['branch'].replace(rename)
            melted['genotype'] = genotype
            melted['side'] = SIDES[sec]
            frames.append(melted)
    return pd.concat(frames, ignore_index=True)
