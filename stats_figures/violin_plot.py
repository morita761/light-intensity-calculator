"""極性の大きさ |L - R| の split violin plot

用途: 左右branchの輝度差の絶対値 |L - R|（= 極性の強さ）が系統間で変わるかを見る。
      1つのバイオリンの左半分が Ventral R8、右半分が Dorsal R8。
      赤線は中央値、点線は四分位。先頭の系統（Control）と他の各系統を t検定し、
      ブラケットとアスタリスクで表示する。
入力: data/processed/<dataset>/*_v.csv, *_d.csv（parse_intensity.py で生成）
出力: output/violin_plot/violin_plot_<dataset>.png（文字無し版は *_notext.png）

Usage:
    python violin_plot.py                          # Control vs Cas9
    python violin_plot.py --dataset three_groups   # Control / vang RNAi / vang Cas9
    python violin_plot.py --no-text                # 文字無し版のみ
    python violin_plot.py --both --no-show         # 通常版+文字無し版を保存、表示しない
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from lib.data_io import abs_diff_long, load_vd
from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings
from lib.stats_utils import p_to_asterisks

NAME = 'violin_plot'


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    data = load_vd(args.dataset)
    order = list(data)
    diff_all = abs_diff_long(data)

    # 統計: 先頭の系統 vs 各系統
    ref = order[0]
    ref_data = diff_all[diff_all['genotype'] == ref]['abs_diff']
    comparisons = []
    for i, genotype in enumerate(order[1:], start=1):
        _, p_val = stats.ttest_ind(ref_data, diff_all[diff_all['genotype'] == genotype]['abs_diff'])
        comparisons.append((i, p_val, p_to_asterisks(p_val)))
        print(f"{ref} vs {genotype}: P = {p_val:.2e} ({p_to_asterisks(p_val)})")

    default_width = 4 + 3 * len(order)
    fig, ax = plt.subplots(figsize=s.floats('figsize', (default_width, 7)))

    sns.violinplot(data=diff_all, x='genotype', y='abs_diff', hue='side', order=order,
                   split=True, inner='quartile', palette=s.str('palette', 'Set2'), alpha=0.7, ax=ax)

    # 各半バイオリンに四分位線が3本ずつあり、0,1,2の「1」が中央値
    for i, line in enumerate(ax.lines):
        if i % 3 == 1:
            line.set_color(s.str('median_color', 'red'))
            line.set_linewidth(s.float('median_linewidth', 2.5))
            line.set_linestyle('-')
            line.set_alpha(1.0)

    sns.stripplot(data=diff_all, x='genotype', y='abs_diff', hue='side', order=order,
                  dodge=True, palette='dark:black', alpha=0.2, jitter=True, size=2, legend=False, ax=ax)

    # ブラケットとアスタリスク（比較ごとに上へずらす）
    max_val = diff_all['abs_diff'].max()
    offset = s.float('bracket_offset', 8)
    step = s.float('bracket_step', 10)
    height = s.float('bracket_height', 1)
    line_y = max_val
    for n, (i, p_val, asterisks) in enumerate(comparisons):
        line_y = max_val + offset + step * n
        ax.plot([0, 0, i, i], [line_y, line_y + height, line_y + height, line_y], lw=1.5, c='black')
        ax.text(i / 2, line_y + height, asterisks, ha='center', va='bottom',
                fontsize=s.float('star_fontsize', 16))

    s.apply(ax)
    if s.ylim() is None:
        ax.set_ylim(0, line_y + s.float('top_margin', 8))
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(s.str('legend_title', '') or None)

    fig.tight_layout()
    save_figure(fig, f'{NAME}_{args.dataset}', args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
