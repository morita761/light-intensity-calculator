"""V-D分岐の箱ひげ図（Ventral R8s / Dorsal R8s を横並び）

用途: 1系統（既定は control.txt）について、Ventral R8 と Dorsal R8 それぞれで
      Ventral branch と Dorsal branch の輝度差を t検定付きの箱ひげ図で示す。
      process_data.py（旧ノーマライズ方式）の出力を使う。
入力: data/processed/single/v_normalize.csv, d_normalize.csv（process_data.py で生成）
出力: output/boxplot_vd/boxplot_vd.png

Usage:
    python process_data.py      # 先にCSVを作る
    python boxplot_vd.py
    python boxplot_vd.py --both
"""
import matplotlib.pyplot as plt

from lib.data_io import read_lr_csv
from lib.figure import finish, make_parser, save_figure
from lib.paths import PROCESSED_DIR
from lib.plots import vd_boxplot
from lib.settings import load_figure_settings

NAME = 'boxplot_vd'


def main():
    args = make_parser(__doc__, NAME).parse_args()
    s = load_figure_settings(NAME)

    fig, axes = plt.subplots(1, 2, figsize=s.floats('figsize', (8, 4)))
    for ax, sec, prefix in zip(axes, ('v', 'd'), ('ventral_', 'dorsal_')):
        df = read_lr_csv(PROCESSED_DIR / 'single' / f'{sec}_normalize.csv')
        _, p_val = vd_boxplot(ax, df, s)
        s.apply(ax, prefix=prefix)
        print(f"{s.str(prefix + 'title')}: p = {p_val:.2e}")

    fig.tight_layout()
    save_figure(fig, NAME, args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
