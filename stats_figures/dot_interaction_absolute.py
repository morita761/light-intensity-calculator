"""個別ドット付きの交互作用プロット + |L - R| 棒グラフ（2パネル）

用途: 左パネルで Ventral branch / Dorsal branch の輝度を系統ごとに個別ドットと
      平均±SEで示し（系統によって左右差の傾きが変わるか）、
      右パネルで |L - R| の平均±SEを個別ドット付きで比較する。
入力: data/processed/<dataset>/*_v.csv, *_d.csv
出力: output/dot_interaction_absolute/dot_interaction_absolute_<dataset>.png

Usage:
    python dot_interaction_absolute.py
    python dot_interaction_absolute.py --dataset three_groups --no-text
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from lib.data_io import abs_diff_long, intensity_long, load_vd
from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings

NAME = 'dot_interaction_absolute'


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    data = load_vd(args.dataset)
    order = list(data)
    data_all = intensity_long(data, s.list('branch_labels'))
    diff_all = abs_diff_long(data)
    palette = s.str('palette', 'Set2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=s.floats('figsize', (14, 6)))

    # A. Interaction Plot (個別ドット + 平均線)
    sns.stripplot(data=data_all, x='branch', y='intensity', hue='genotype', hue_order=order,
                  dodge=True, alpha=0.3, palette=palette, ax=ax1, jitter=True, size=3)
    sns.pointplot(data=data_all, x='branch', y='intensity', hue='genotype', hue_order=order,
                  dodge=0.4, capsize=.1, markers="D", linestyles="-", ax=ax1, errorbar='se',
                  palette='dark:black')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:len(order)], labels[:len(order)], title=s.str('legend_title', 'Genotype'))
    s.apply(ax1, prefix='left_')

    # B. Absolute Difference Bar Plot (個別ドット + 棒グラフ)
    sns.barplot(data=diff_all, x='genotype', y='abs_diff', order=order, hue='genotype',
                capsize=.1, palette=palette, ax=ax2, errorbar='se', alpha=0.6, legend=False)
    sns.stripplot(data=diff_all, x='genotype', y='abs_diff', order=order,
                  alpha=0.5, color='black', ax=ax2, jitter=True, size=3)
    s.apply(ax2, prefix='right_')

    ref_data = diff_all[diff_all['genotype'] == order[0]]['abs_diff']
    for genotype in order[1:]:
        _, p_val = stats.ttest_ind(ref_data, diff_all[diff_all['genotype'] == genotype]['abs_diff'])
        print(f"|L - R| {order[0]} vs {genotype}: T-test P = {p_val:.2e}")

    fig.tight_layout()
    save_figure(fig, f'{NAME}_{args.dataset}', args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
