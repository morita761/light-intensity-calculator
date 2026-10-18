"""Two-way ANOVA: Experiment（系統）× Branch（Ventral/Dorsal branch）

用途: 「KOで Ventral/Dorsal branch 間の輝度差が緩和されるか」を検定する詳細版。
      各CSVの left_intensity = Ventral branch, right_intensity = Dorsal branch として、
      記述統計 / 条件別 t検定 / 全体・Side別 ANOVA（効果量η²付き）/ Tukey HSD を
      コンソールに出力し、図を4種類保存する。
入力: data/processed/<dataset>/*_v.csv, *_d.csv
出力: output/twoway_anova/twoway_anova_<dataset>_*.png
        _interaction        交互作用プロット（全データ）
        _boxplot_conditions 条件別箱ひげ図（t検定付き）
        _branch_difference  branch差（V - D）の比較
        _summary            Side別のまとめ（メイン結果）

Usage:
    python twoway_anova.py
    python twoway_anova.py --dataset three_groups --both --no-show
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from lib.data_io import intensity_long, load_vd
from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings
from lib.stats_utils import p_to_asterisks

NAME = 'twoway_anova'
BRANCHES = ['Ventral', 'Dorsal']
SIDES = ['Ventral', 'Dorsal']


def load_and_prepare_data(dataset):
    """縦持ちデータ (experiment, side, branch, intensity) を作る"""
    df = intensity_long(load_vd(dataset), BRANCHES)
    return df.rename(columns={'genotype': 'experiment'})


def print_descriptive_stats(df):
    print("\n" + "=" * 70)
    print("記述統計量")
    print("=" * 70)
    grouped = df.groupby(['experiment', 'side', 'branch'], sort=False)['intensity']
    print(grouped.agg(['count', 'mean', 'std', 'median']).round(3))


def run_ttest_branch_comparison(df, experiments):
    """各条件での Ventral branch vs Dorsal branch の t検定"""
    print("\n" + "=" * 70)
    print("t検定: Ventral branch vs Dorsal branch (各条件)")
    print("=" * 70)

    results = []
    for experiment in experiments:
        for side in SIDES:
            df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]
            ventral = df_subset[df_subset['branch'] == 'Ventral']['intensity']
            dorsal = df_subset[df_subset['branch'] == 'Dorsal']['intensity']

            t_stat, p_val = stats.ttest_ind(ventral, dorsal)
            sig = p_to_asterisks(p_val)
            diff = ventral.mean() - dorsal.mean()

            print(f"\n【{experiment} - {side} side】")
            print(f"  Ventral branch: mean = {ventral.mean():.2f}, n = {len(ventral)}")
            print(f"  Dorsal branch:  mean = {dorsal.mean():.2f}, n = {len(dorsal)}")
            print(f"  差 (V - D): {diff:.2f}")
            print(f"  t = {t_stat:.4f}, p = {p_val:.6f} {sig}")

            results.append({'experiment': experiment, 'side': side,
                            'ventral_mean': ventral.mean(), 'dorsal_mean': dorsal.mean(),
                            'diff': diff, 't_stat': t_stat, 'p_val': p_val, 'sig': sig})
    return pd.DataFrame(results)


def run_anova(df, label):
    model = ols('intensity ~ C(experiment) * C(branch)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\n【ANOVA Table】{label}")
    print(anova_table)

    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    print("\n【Effect Size (η²) / 有意性判定】")
    for idx in anova_table.index[:-1]:
        p_val = anova_table.loc[idx, 'PR(>F)']
        print(f"  {idx}: η² = {anova_table.loc[idx, 'eta_sq']:.4f}, p = {p_val:.6f} {p_to_asterisks(p_val)}")
    return anova_table


def run_posthoc_tukey(df):
    print("\n" + "=" * 70)
    print("事後検定 (Tukey HSD): Experiment × Branch")
    print("=" * 70)
    groups = df['experiment'] + '_' + df['branch']
    tukey = pairwise_tukeyhsd(df['intensity'], groups, alpha=0.05)
    print(tukey)
    return tukey


def mean_se(df, keys):
    summary = df.groupby(keys, sort=False)['intensity'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    return summary.reset_index()


def errorbar_lines(ax, summary, x, line, line_values, colors, label_fmt='{}'):
    for value in line_values:
        data = summary[summary[line] == value]
        ax.errorbar(data[x], data['mean'], yerr=data['se'],
                    marker='o', markersize=10, capsize=5, capthick=2, linewidth=2.5,
                    label=label_fmt.format(value), color=colors[value])


def plot_interaction_exp_branch(df, anova_table, experiments, colors, s, args, tag):
    """Experiment × Branch の交互作用プロット（全データ）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=s.floats('interaction_figsize', (12, 5)))
    summary = mean_se(df, ['experiment', 'branch'])

    errorbar_lines(ax1, summary, 'experiment', 'branch', BRANCHES, colors['branch'], '{} branch')
    s.apply(ax1, prefix='interaction_left_')
    ax1.legend(fontsize=s.float('legend_fontsize'))

    errorbar_lines(ax2, summary, 'branch', 'experiment', experiments, colors['experiment'])
    s.apply(ax2, prefix='interaction_right_')
    ax2.legend(fontsize=s.float('legend_fontsize'))

    p_interaction = anova_table.loc['C(experiment):C(branch)', 'PR(>F)']
    fig.suptitle(f"{s.str('interaction_suptitle')}\nInteraction p = {p_interaction:.2e} "
                 f"{p_to_asterisks(p_interaction)}", fontsize=14, y=1.05)
    fig.tight_layout()
    save_figure(fig, f'{tag}_interaction', args, s, NAME)


def plot_boxplot_by_condition(df, ttest_results, experiments, colors, s, args, tag):
    """条件別の箱ひげ図（t検定結果付き）"""
    n_rows = len(experiments)
    fig, axes = plt.subplots(n_rows, 2, figsize=s.floats('conditions_figsize', (12, 5 * n_rows)),
                             squeeze=False)

    for row, experiment in enumerate(experiments):
        for col, side in enumerate(SIDES):
            ax = axes[row, col]
            df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]
            data_v = df_subset[df_subset['branch'] == 'Ventral']['intensity']
            data_d = df_subset[df_subset['branch'] == 'Dorsal']['intensity']

            box = ax.boxplot([data_v, data_d], tick_labels=['Ventral\nbranch', 'Dorsal\nbranch'],
                             patch_artist=True, widths=0.6)
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors['branch'][BRANCHES[i]])
                patch.set_alpha(0.7)

            ax.text(1, data_v.max() + 2, f'n={len(data_v)}', ha='center', fontsize=11)
            ax.text(2, data_d.max() + 2, f'n={len(data_d)}', ha='center', fontsize=11)

            result = ttest_results[(ttest_results['experiment'] == experiment) &
                                   (ttest_results['side'] == side)].iloc[0]
            y_max = max(data_v.max(), data_d.max()) + 5
            ax.plot([1, 1, 2, 2], [y_max, y_max + 2, y_max + 2, y_max], lw=1.5, c='black')
            ax.text(1.5, y_max + 2, result['sig'], ha='center', fontsize=16, fontweight='bold')

            s.apply(ax, prefix='conditions_')
            ax.set_title(f'{experiment} - {side} side\n(p = {result["p_val"]:.2e})',
                         fontsize=s.float('title_fontsize'))
            if s.ylim('conditions_ylim') is None:
                ax.set_ylim(0, y_max + 15)

    fig.suptitle(s.str('conditions_suptitle'), fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, f'{tag}_boxplot_conditions', args, s, NAME)


def plot_branch_difference(df, experiments, colors, s, args, tag):
    """Branch間の差（Ventral - Dorsal）を条件ごとに比較"""
    fig, ax = plt.subplots(figsize=s.floats('branch_difference_figsize', (10, 6)))

    groups, diffs, box_colors = [], [], []
    for experiment in experiments:
        for side in SIDES:
            df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]
            ventral = df_subset[df_subset['branch'] == 'Ventral']['intensity'].values
            dorsal = df_subset[df_subset['branch'] == 'Dorsal']['intensity'].values
            # 同じ行（同じ細胞）同士の差
            min_len = min(len(ventral), len(dorsal))
            groups.append(f'{experiment}\n{side} side')
            diffs.append(ventral[:min_len] - dorsal[:min_len])
            box_colors.append(colors['experiment'][experiment])

    box = ax.boxplot(diffs, tick_labels=groups, patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    s.apply(ax, prefix='branch_difference_')

    for i, d in enumerate(diffs):
        ax.text(i + 1, ax.get_ylim()[1] * 0.9, f'mean={d.mean():.1f}', ha='center', fontsize=10)

    fig.tight_layout()
    save_figure(fig, f'{tag}_branch_difference', args, s, NAME)


def plot_summary_by_side(df, anova_by_side, ttest_results, experiments, colors, s, args, tag):
    """Side別のサマリー図（メイン結果）"""
    fig = plt.figure(figsize=s.floats('summary_figsize', (14, 10)))

    # 上段: Side別の交互作用プロット
    p_int = {}
    for i, side in enumerate(SIDES, start=1):
        ax = fig.add_subplot(2, 2, i)
        summary = mean_se(df[df['side'] == side], ['experiment', 'branch'])
        errorbar_lines(ax, summary, 'experiment', 'branch', BRANCHES, colors['branch'], '{} branch')
        p_int[side] = anova_by_side[side].loc['C(experiment):C(branch)', 'PR(>F)']
        s.apply(ax, prefix='summary_interaction_')
        ax.set_title(f'{side} side\nInteraction: p = {p_int[side]:.2e} {p_to_asterisks(p_int[side])}',
                     fontsize=s.float('title_fontsize'))
        ax.legend(fontsize=s.float('legend_fontsize'))

    # 左下: Branch差の棒グラフ
    ax3 = fig.add_subplot(2, 2, 3)
    x_labels = [f"{r['experiment']}\n{r['side']}" for _, r in ttest_results.iterrows()]
    bars = ax3.bar(x_labels, ttest_results['diff'].abs(),
                   color=[colors['experiment'][e] for e in ttest_results['experiment']],
                   alpha=0.7, edgecolor='black')
    for bar, sig in zip(bars, ttest_results['sig']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 sig, ha='center', fontsize=14, fontweight='bold')
    s.apply(ax3, prefix='summary_bar_')

    # 右下: 結果サマリー（テキスト）
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    lines = ["Two-way ANOVA Results (by side)", "=" * 45, ""]
    for side in SIDES:
        lines += [f"{side} side (Experiment x Branch):",
                  f"  Interaction: p = {p_int[side]:.2e} {p_to_asterisks(p_int[side])}", ""]
    lines += ["=" * 45, "t-test: V branch vs D branch", "-" * 45]
    for _, row in ttest_results.iterrows():
        lines.append(f"{row['experiment']:8} {row['side']:8}: "
                     f"diff={row['diff']:+.1f}, p={row['p_val']:.1e} {row['sig']}")
    conclusion = s.str('summary_conclusion')
    if conclusion:
        lines += ["", "=" * 45, "Conclusion:", conclusion]
    ax4.text(0.05, 0.95, '\n'.join(lines), fontsize=11, family='monospace',
             transform=ax4.transAxes, verticalalignment='top')

    fig.suptitle(s.str('summary_suptitle'), fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, f'{tag}_summary', args, s, NAME)


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    df = load_and_prepare_data(args.dataset)
    experiments = list(df['experiment'].unique())
    group_colors = s.list('group_colors', ['#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    colors = {
        'branch': dict(zip(BRANCHES, s.list('branch_colors', ['#1f77b4', '#ff7f0e']))),
        'experiment': dict(zip(experiments, group_colors)),
    }
    print(f"総データ数: {len(df)}")

    print_descriptive_stats(df)
    ttest_results = run_ttest_branch_comparison(df, experiments)

    print("\n" + "=" * 70)
    print(f"Two-way ANOVA: Experiment ({'/'.join(experiments)}) × Branch (Ventral/Dorsal)")
    print("=" * 70)
    anova_table = run_anova(df, '(全データ)')

    anova_by_side = {}
    for side in SIDES:
        print("\n" + "=" * 70)
        print(f"Two-way ANOVA ({side} side のみ): Experiment × Branch")
        print("=" * 70)
        anova_by_side[side] = run_anova(df[df['side'] == side], f'({side} side)')

    run_posthoc_tukey(df)

    print("\n" + "=" * 70)
    print("図の生成")
    print("=" * 70)
    tag = f'{NAME}_{args.dataset}'
    plot_interaction_exp_branch(df, anova_table, experiments, colors, s, args, tag)
    plot_boxplot_by_condition(df, ttest_results, experiments, colors, s, args, tag)
    plot_branch_difference(df, experiments, colors, s, args, tag)
    plot_summary_by_side(df, anova_by_side, ttest_results, experiments, colors, s, args, tag)
    finish(args)


if __name__ == '__main__':
    main()
