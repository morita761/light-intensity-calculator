"""V/Dをまとめた |L - R| の系統間比較（棒グラフ）

用途: Ventral/Dorsal を区別せず、M1層全体として極性（|L - R|）が
      系統間で弱まっているかを見る。棒は平均値、エラーバーは95%信頼区間。
      先頭の系統との t検定の P値をコンソールに表示する。
入力: data/processed/<dataset>/*_v.csv, *_d.csv
出力: output/absolute_diff/absolute_diff_<dataset>.png

Usage:
    python absolute_diff.py
    python absolute_diff.py --dataset three_groups --both
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from lib.data_io import abs_diff_long, load_vd
from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings

NAME = 'absolute_diff'


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    data = load_vd(args.dataset)
    order = list(data)
    diff_all = abs_diff_long(data)

    # V/D をまとめた状態で、先頭の系統 vs 各系統の t検定
    ref = order[0]
    ref_data = diff_all[diff_all['genotype'] == ref]['abs_diff']
    print(f"{ref} Mean: {ref_data.mean():.2f}")
    for genotype in order[1:]:
        other = diff_all[diff_all['genotype'] == genotype]['abs_diff']
        _, p_val = stats.ttest_ind(ref_data, other)
        print(f"{genotype} Mean: {other.mean():.2f}  (Pooled t-test vs {ref}: P = {p_val:.2e})")

    fig, ax = plt.subplots(figsize=s.floats('figsize', (6, 6)))
    sns.barplot(data=diff_all, x='genotype', y='abs_diff', order=order, hue='genotype',
                capsize=.1, palette=s.str('palette', 'Set2'), legend=False, ax=ax)
    s.apply(ax)

    fig.tight_layout()
    save_figure(fig, f'{NAME}_{args.dataset}', args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
