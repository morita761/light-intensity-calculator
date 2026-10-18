"""Side別 (Ventral / Dorsal) Two-way ANOVA: 系統 (genotype) × 部位 (branch)

用途: Ventral R8 と Dorsal R8 を分けて、系統と部位の主効果・交互作用を検定し、
      Tukey HSD で系統間の多重比較を行う。結果を表画像としても保存する
      （発表・報告用）。twoway_anova.py より出力が簡潔。
入力: data/processed/<dataset>/*_v.csv, *_d.csv
出力: output/twoway_anova_genotype_x_branch/twoway_anova_genotype_x_branch_<dataset>_*.png
        _anova_table  ANOVA結果の表（文字無し版なし）
        _tukey_table  Tukey HSD結果の表（文字無し版なし）
        _interaction  交互作用プロット
        _boxplot      箱ひげ図

Usage:
    python twoway_anova_genotype_x_branch.py
    python twoway_anova_genotype_x_branch.py --dataset three_groups --both
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from lib.data_io import intensity_long, load_vd
from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings
from lib.stats_utils import format_p_scientific

# 表の日本語表示用フォント
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Yu Gothic', 'Meiryo', 'IPAGothic',
                                      'Noto Sans CJK JP', 'Hiragino Sans', 'sans-serif']
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

NAME = 'twoway_anova_genotype_x_branch'
SIDE_COLORS = {'Ventral': ('#E6F3FF', '#E6FFE6'), 'Dorsal': ('#FFF2E6', '#FFFFE6')}


def draw_table(df_display, title, header_color, row_color_index, figsize, fontsize, scale_y):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    table = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                     cellLoc='center', loc='upper center',
                     colColours=[header_color] * len(df_display.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale_y)
    for j in range(len(df_display.columns)):
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # Ventral/Dorsalで背景色を分ける
    for i, side in enumerate(df_display['解析対象']):
        for j in range(len(df_display.columns)):
            table[(i + 1, j)].set_facecolor(SIDE_COLORS[side][row_color_index])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=5)
    fig.tight_layout()
    return fig


def anova_table_figure(anova_by_side, title):
    label_map = {
        'C(genotype)': '系統 (Genotype)',
        'C(branch)': '部位 (Branch)',
        'C(genotype):C(branch)': '交互作用 (系統×部位)',
    }
    rows = []
    for side, anova in anova_by_side.items():
        for idx, row in anova.drop(index='Residual').iterrows():
            rows.append({'解析対象': side, '要因': label_map.get(idx, idx), 'df': f"{int(row['df'])}",
                         'F': f"{row['F']:.2f}", 'P値': format_p_scientific(row['PR(>F)'])})
    return draw_table(pd.DataFrame(rows), title, '#4472C4', 0, (10, 4), 13, 1.8)


def tukey_table_figure(tukey_by_side, title):
    rows = []
    for side, tukey in tukey_by_side.items():
        table = tukey._results_table.data
        for r in table[1:]:
            rec = dict(zip(table[0], r))
            rows.append({'解析対象': side, '比較1': rec['group1'], '比較2': rec['group2'],
                         '平均差': f"{float(rec['meandiff']):.3f}",
                         '調整済P値': format_p_scientific(float(rec['p-adj'])),
                         '有意': 'Yes' if rec['reject'] else 'No'})
    return draw_table(pd.DataFrame(rows), title, '#70AD47', 1, (11, 4.5), 12, 1.6)


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    data = load_vd(args.dataset)
    order = list(data)
    df = intensity_long(data, s.list('branch_labels'))
    by_side = {side: df[df['side'] == side] for side in ('Ventral', 'Dorsal')}

    # 1. Two-way ANOVA
    print("=" * 60)
    print(f"Two-way ANOVA Results ({' / '.join(order)})")
    print("=" * 60)
    anova_by_side = {}
    for side, df_side in by_side.items():
        print(f"\n[{side} Side]")
        model = ols('intensity ~ C(genotype) * C(branch)', data=df_side).fit()
        anova_by_side[side] = sm.stats.anova_lm(model, typ=2)
        print(anova_by_side[side])

    print("\n" + "=" * 60)
    print("解釈:")
    print(f"- C(genotype): 系統間（{' vs '.join(order)}）の主効果")
    print("- C(branch): 部位間（Ventral branch vs Dorsal branch）の主効果")
    print("- C(genotype):C(branch): 交互作用（系統によって左右差が異なるか）")
    print("=" * 60)

    # 2. 事後検定 (Tukey HSD)
    print("\n" + "=" * 60)
    print("Post-hoc Test (Tukey HSD) - Genotype Comparison")
    print("=" * 60)
    tukey_by_side = {}
    for side, df_side in by_side.items():
        print(f"\n[{side} Side]")
        tukey_by_side[side] = pairwise_tukeyhsd(df_side['intensity'], df_side['genotype'], alpha=0.05)
        print(tukey_by_side[side])

    tag = f'{NAME}_{args.dataset}'
    # 表は文字そのものが内容なので文字無し版は作らない
    fig = anova_table_figure(anova_by_side, s.str('anova_table_title', 'Two-way ANOVA Results'))
    save_figure(fig, f'{tag}_anova_table', args, s, NAME, text_free=False)
    fig = tukey_table_figure(tukey_by_side, s.str('tukey_table_title', 'Tukey HSD 多重比較結果'))
    save_figure(fig, f'{tag}_tukey_table', args, s, NAME, text_free=False)

    # 3. 交互作用プロット
    fig, axes = plt.subplots(1, 2, figsize=s.floats('figsize', (14, 5)))
    for ax, (side, df_side) in zip(axes, by_side.items()):
        sns.pointplot(data=df_side, x='branch', y='intensity', hue='genotype',
                      hue_order=order, ax=ax, capsize=.1)
        ax.legend(title=s.str('legend_title', 'Genotype'))
        s.apply(ax, prefix=(f'interaction_{side.lower()}_', 'interaction_'))
    fig.tight_layout()
    save_figure(fig, f'{tag}_interaction', args, s, NAME)

    # 4. 箱ひげ図
    fig, axes = plt.subplots(1, 2, figsize=s.floats('figsize', (14, 5)))
    for ax, (side, df_side) in zip(axes, by_side.items()):
        sns.boxplot(data=df_side, x='genotype', y='intensity', hue='branch', order=order, ax=ax)
        s.apply(ax, prefix=(f'boxplot_{side.lower()}_', 'boxplot_'))
    fig.tight_layout()
    save_figure(fig, f'{tag}_boxplot', args, s, NAME)

    finish(args)


if __name__ == '__main__':
    main()
