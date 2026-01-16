import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """
    4つのCSVファイルを読み込み、Two-way ANOVA用のデータフレームを作成する。
    要因1: experiment (control / cas9)
    要因2: section (V / D)
    従属変数: intensity (left_intensity と right_intensity を統合)
    """
    files = {
        ('control', 'V'): 'control_v.csv',
        ('control', 'D'): 'control_d.csv',
        ('cas9', 'V'): 'cas9_v.csv',
        ('cas9', 'D'): 'cas9_d.csv',
    }

    all_data = []

    for (experiment, section), filename in files.items():
        try:
            df = pd.read_csv(filename, sep=r'\s+')

            # left_intensity (Ventral branch) のデータ
            for val in df['left_intensity']:
                all_data.append({
                    'experiment': experiment,
                    'section': section,
                    'branch': 'Ventral',
                    'intensity': val
                })

            # right_intensity (Dorsal branch) のデータ
            for val in df['right_intensity']:
                all_data.append({
                    'experiment': experiment,
                    'section': section,
                    'branch': 'Dorsal',
                    'intensity': val
                })

        except FileNotFoundError:
            print(f"エラー: ファイル '{filename}' が見つかりません。")
            return None

    return pd.DataFrame(all_data)


def run_twoway_anova(df):
    """
    Two-way ANOVAを実行する。
    要因: experiment (control/cas9) x section (V/D)
    """
    print("=" * 60)
    print("Two-way ANOVA: Experiment × Section")
    print("=" * 60)

    # statsmodelsを使用したTwo-way ANOVA
    model = ols('intensity ~ C(experiment) * C(section)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA Table】")
    print(anova_table)

    # 効果量 (η²) の計算
    ss_total = anova_table['sum_sq'].sum()
    anova_table['eta_sq'] = anova_table['sum_sq'] / ss_total

    print("\n【Effect Size (η²)】")
    for idx in anova_table.index[:-1]:  # Residual以外
        print(f"  {idx}: η² = {anova_table.loc[idx, 'eta_sq']:.4f}")

    # 有意性の判定
    print("\n【有意性判定】")
    for idx in anova_table.index[:-1]:
        p_val = anova_table.loc[idx, 'PR(>F)']
        sig = get_significance(p_val)
        print(f"  {idx}: p = {p_val:.6f} {sig}")

    return anova_table


def run_twoway_anova_with_branch(df):
    """
    Three-way ANOVAを実行する（branchを含む場合）。
    要因: experiment (control/cas9) x section (V/D) x branch (Ventral/Dorsal)
    """
    print("\n" + "=" * 60)
    print("Three-way ANOVA: Experiment × Section × Branch")
    print("=" * 60)

    model = ols('intensity ~ C(experiment) * C(section) * C(branch)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA Table】")
    print(anova_table)

    # 有意性の判定
    print("\n【有意性判定】")
    for idx in anova_table.index[:-1]:
        p_val = anova_table.loc[idx, 'PR(>F)']
        sig = get_significance(p_val)
        print(f"  {idx}: p = {p_val:.6f} {sig}")

    return anova_table


def run_simple_twoway_anova(df):
    """
    シンプルなTwo-way ANOVA（left_intensityのみ使用）。
    要因: experiment (control/cas9) x section (V/D)
    """
    print("\n" + "=" * 60)
    print("Two-way ANOVA (Ventral branch のみ): Experiment × Section")
    print("=" * 60)

    # Ventral branchのデータのみを使用
    df_ventral = df[df['branch'] == 'Ventral'].copy()

    model = ols('intensity ~ C(experiment) * C(section)', data=df_ventral).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA Table】")
    print(anova_table)

    # 有意性の判定
    print("\n【有意性判定】")
    for idx in anova_table.index[:-1]:
        p_val = anova_table.loc[idx, 'PR(>F)']
        sig = get_significance(p_val)
        print(f"  {idx}: p = {p_val:.6f} {sig}")

    return anova_table


def get_significance(p):
    """p値を有意性マークに変換"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def print_descriptive_stats(df):
    """記述統計量を表示"""
    print("\n" + "=" * 60)
    print("記述統計量")
    print("=" * 60)

    grouped = df.groupby(['experiment', 'section', 'branch'])['intensity']
    stats_df = grouped.agg(['count', 'mean', 'std', 'median']).round(3)
    print(stats_df)

    # グループ別（experiment × section）の要約
    print("\n【Experiment × Section 別の要約】")
    grouped2 = df.groupby(['experiment', 'section'])['intensity']
    stats_df2 = grouped2.agg(['count', 'mean', 'std']).round(3)
    print(stats_df2)


def run_posthoc_tukey(df):
    """Tukey HSD事後検定"""
    print("\n" + "=" * 60)
    print("事後検定 (Tukey HSD)")
    print("=" * 60)

    # experiment × section の組み合わせでグループを作成
    df_copy = df.copy()
    df_copy['group'] = df_copy['experiment'] + '_' + df_copy['section']

    tukey = pairwise_tukeyhsd(df_copy['intensity'], df_copy['group'], alpha=0.05)
    print(tukey)
    return tukey


def plot_interaction(df, anova_table):
    """交互作用プロットを描画"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 平均値と標準誤差を計算
    summary = df.groupby(['experiment', 'section'])['intensity'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.reset_index()

    # --- 左図: Experiment を x 軸、Section で線を分ける ---
    ax1 = axes[0]
    for section in ['V', 'D']:
        data = summary[summary['section'] == section]
        ax1.errorbar(
            data['experiment'], data['mean'], yerr=data['se'],
            marker='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label=f'Section {section}'
        )
    ax1.set_xlabel('Experiment', fontsize=14)
    ax1.set_ylabel('Intensity (Mean ± SE)', fontsize=14)
    ax1.set_title('Interaction Plot\n(X: Experiment, Lines: Section)', fontsize=13)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)

    # --- 右図: Section を x 軸、Experiment で線を分ける ---
    ax2 = axes[1]
    for experiment in ['control', 'cas9']:
        data = summary[summary['experiment'] == experiment]
        ax2.errorbar(
            data['section'], data['mean'], yerr=data['se'],
            marker='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label=experiment
        )
    ax2.set_xlabel('Section', fontsize=14)
    ax2.set_ylabel('Intensity (Mean ± SE)', fontsize=14)
    ax2.set_title('Interaction Plot\n(X: Section, Lines: Experiment)', fontsize=13)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)

    # ANOVA結果をテキストで追加
    p_interaction = anova_table.loc['C(experiment):C(section)', 'PR(>F)']
    sig = get_significance(p_interaction)
    fig.suptitle(f'Two-way ANOVA: Interaction p = {p_interaction:.2e} {sig}', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig('twoway_anova_interaction.png', dpi=150, bbox_inches='tight')
    print("\n交互作用プロットを 'twoway_anova_interaction.png' に保存しました。")
    plt.show()


def plot_boxplot_grouped(df, anova_table):
    """グループ別箱ひげ図を描画"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # グループラベルを作成
    df_plot = df.copy()
    df_plot['group'] = df_plot['experiment'] + '\n' + df_plot['section']

    # 箱ひげ図
    order = ['control\nV', 'control\nD', 'cas9\nV', 'cas9\nD']
    box = sns.boxplot(
        x='group', y='intensity', data=df_plot,
        order=order, hue='group', palette='Set2', legend=False, ax=ax
    )

    # n数を表示
    group_counts = df_plot.groupby('group').size()
    for i, grp in enumerate(order):
        n = group_counts.get(grp, 0)
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=11)

    ax.set_xlabel('Group (Experiment / Section)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.tick_params(labelsize=12)

    # ANOVA結果をタイトルに
    p_exp = anova_table.loc['C(experiment)', 'PR(>F)']
    p_sec = anova_table.loc['C(section)', 'PR(>F)']
    p_int = anova_table.loc['C(experiment):C(section)', 'PR(>F)']

    title = (f'Two-way ANOVA Results\n'
             f'Experiment: p={p_exp:.3f} {get_significance(p_exp)}, '
             f'Section: p={p_sec:.3f} {get_significance(p_sec)}, '
             f'Interaction: p={p_int:.2e} {get_significance(p_int)}')
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig('twoway_anova_boxplot.png', dpi=150, bbox_inches='tight')
    print("箱ひげ図を 'twoway_anova_boxplot.png' に保存しました。")
    plt.show()


def plot_tukey_result(tukey):
    """Tukey HSD結果を可視化"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Tukey結果をDataFrameに変換
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0])

    # 有意な組み合わせを色分け
    colors = ['green' if r else 'gray' for r in tukey_df['reject']]

    y_pos = range(len(tukey_df))
    ax.barh(y_pos, tukey_df['meandiff'], color=colors, edgecolor='black')

    # ラベル
    labels = [f"{row['group1']} vs {row['group2']}" for _, row in tukey_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)

    # 0線
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel('Mean Difference', fontsize=14)
    ax.set_title('Tukey HSD Post-hoc Test\n(Green: p < 0.05, Gray: n.s.)', fontsize=13)
    ax.tick_params(labelsize=11)

    # p値を表示
    for i, (_, row) in enumerate(tukey_df.iterrows()):
        p_adj = row['p-adj']
        sig = '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else ''))
        ax.text(row['meandiff'] + 0.3, i, f'p={p_adj:.4f} {sig}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('twoway_anova_tukey.png', dpi=150, bbox_inches='tight')
    print("Tukey HSD結果を 'twoway_anova_tukey.png' に保存しました。")
    plt.show()


def plot_all_results(df, anova_table, tukey):
    """全ての結果を1つの図にまとめる"""
    fig = plt.figure(figsize=(14, 10))

    # --- 1. 交互作用プロット (左上) ---
    ax1 = fig.add_subplot(2, 2, 1)
    summary = df.groupby(['experiment', 'section'])['intensity'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.reset_index()

    for section in ['V', 'D']:
        data = summary[summary['section'] == section]
        ax1.errorbar(
            data['experiment'], data['mean'], yerr=data['se'],
            marker='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label=f'Section {section}'
        )
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Intensity (Mean ± SE)', fontsize=12)
    ax1.set_title('Interaction Plot', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)

    # --- 2. 箱ひげ図 (右上) ---
    ax2 = fig.add_subplot(2, 2, 2)
    df_plot = df.copy()
    df_plot['group'] = df_plot['experiment'] + '\n' + df_plot['section']
    order = ['control\nV', 'control\nD', 'cas9\nV', 'cas9\nD']
    sns.boxplot(x='group', y='intensity', data=df_plot, order=order, hue='group', palette='Set2', legend=False, ax=ax2)
    ax2.set_xlabel('Group', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Box Plot by Group', fontsize=13)
    ax2.tick_params(labelsize=10)

    # --- 3. ANOVA Table (左下) ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    # ANOVAテーブルを整形
    anova_display = anova_table[['sum_sq', 'df', 'F', 'PR(>F)']].copy()
    anova_display.columns = ['SS', 'df', 'F', 'p-value']
    anova_display = anova_display.round(4)

    # 有意性マーク追加
    sig_marks = []
    for idx in anova_display.index:
        if idx == 'Residual':
            sig_marks.append('')
        else:
            sig_marks.append(get_significance(anova_display.loc[idx, 'p-value']))
    anova_display['Sig.'] = sig_marks

    table_text = anova_display.to_string()
    ax3.text(0.1, 0.9, 'Two-way ANOVA Table', fontsize=14, fontweight='bold',
             transform=ax3.transAxes, verticalalignment='top')
    ax3.text(0.1, 0.75, table_text, fontsize=10, family='monospace',
             transform=ax3.transAxes, verticalalignment='top')

    # --- 4. Tukey HSD (右下) ---
    ax4 = fig.add_subplot(2, 2, 4)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0])
    colors = ['green' if r else 'gray' for r in tukey_df['reject']]
    y_pos = range(len(tukey_df))
    ax4.barh(y_pos, tukey_df['meandiff'], color=colors, edgecolor='black')
    labels = [f"{row['group1']} vs {row['group2']}" for _, row in tukey_df.iterrows()]
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Mean Difference', fontsize=12)
    ax4.set_title('Tukey HSD Post-hoc\n(Green: p<0.05)', fontsize=13)
    ax4.tick_params(labelsize=9)

    plt.suptitle('Two-way ANOVA: Experiment × Section', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('twoway_anova_summary.png', dpi=150, bbox_inches='tight')
    print("\n全結果のサマリー図を 'twoway_anova_summary.png' に保存しました。")
    plt.show()


# --- メイン実行部分 ---
if __name__ == '__main__':
    # データの読み込み
    df = load_and_prepare_data()

    if df is None:
        print("データの読み込みに失敗しました。")
        exit(1)

    print(f"総データ数: {len(df)}")

    # 記述統計量
    print_descriptive_stats(df)

    # Two-way ANOVA (experiment × section、全データ使用)
    anova_table = run_twoway_anova(df)

    # Two-way ANOVA (Ventral branchのみ)
    run_simple_twoway_anova(df)

    # Three-way ANOVA (branch を含む)
    run_twoway_anova_with_branch(df)

    # 事後検定
    tukey = run_posthoc_tukey(df)

    # --- 図の表示 ---
    print("\n" + "=" * 60)
    print("図の生成")
    print("=" * 60)

    # 交互作用プロット
    plot_interaction(df, anova_table)

    # 箱ひげ図
    plot_boxplot_grouped(df, anova_table)

    # Tukey HSD結果
    plot_tukey_result(tukey)

    # サマリー図（全結果をまとめた図）
    plot_all_results(df, anova_table, tukey)
