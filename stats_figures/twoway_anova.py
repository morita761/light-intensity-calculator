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

    ファイル構造:
    - control_v.csv, control_d.csv: Control群（Ventral side, Dorsal side）
    - cas9_v.csv, cas9_d.csv: KO群（Ventral side, Dorsal side）

    各ファイル内:
    - left_intensity: Ventral branch の輝度
    - right_intensity: Dorsal branch の輝度

    Two-way ANOVAの要因:
    - experiment: control / KO
    - branch: Ventral / Dorsal
    """
    files = {
        ('control', 'Ventral'): 'control_v.csv',
        ('control', 'Dorsal'): 'control_d.csv',
        ('KO', 'Ventral'): 'cas9_v.csv',
        ('KO', 'Dorsal'): 'cas9_d.csv',
    }

    all_data = []

    for (experiment, side), filename in files.items():
        try:
            df = pd.read_csv(filename, sep=r'\s+')

            # left_intensity = Ventral branch
            for val in df['left_intensity']:
                all_data.append({
                    'experiment': experiment,
                    'side': side,
                    'branch': 'Ventral',
                    'intensity': val
                })

            # right_intensity = Dorsal branch
            for val in df['right_intensity']:
                all_data.append({
                    'experiment': experiment,
                    'side': side,
                    'branch': 'Dorsal',
                    'intensity': val
                })

        except FileNotFoundError:
            print(f"エラー: ファイル '{filename}' が見つかりません。")
            return None

    return pd.DataFrame(all_data)


def get_significance(p):
    """p値を有意性マークに変換"""
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def run_twoway_anova_exp_branch(df):
    """
    Two-way ANOVA: Experiment × Branch

    目的: KOによってVentral/Dorsal branch間の輝度差が緩和されるかを検定
    """
    print("=" * 70)
    print("Two-way ANOVA: Experiment (control/KO) × Branch (Ventral/Dorsal)")
    print("=" * 70)
    print("\n【仮説】")
    print("  - Control群: Ventral branch と Dorsal branch で大きな輝度差がある")
    print("  - KO群: その差が緩和される")
    print("  → Experiment × Branch の交互作用が有意であることを期待")

    model = ols('intensity ~ C(experiment) * C(branch)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA Table】")
    print(anova_table)

    # 効果量 (η²)
    ss_total = anova_table['sum_sq'].sum()
    anova_table['eta_sq'] = anova_table['sum_sq'] / ss_total

    print("\n【Effect Size (η²)】")
    for idx in anova_table.index[:-1]:
        print(f"  {idx}: η² = {anova_table.loc[idx, 'eta_sq']:.4f}")

    print("\n【有意性判定】")
    for idx in anova_table.index[:-1]:
        p_val = anova_table.loc[idx, 'PR(>F)']
        sig = get_significance(p_val)
        print(f"  {idx}: p = {p_val:.6f} {sig}")

    return anova_table


def run_twoway_anova_by_side(df):
    """
    Side別 (Ventral side / Dorsal side) に Two-way ANOVA を実行
    """
    results = {}

    for side in ['Ventral', 'Dorsal']:
        print("\n" + "=" * 70)
        print(f"Two-way ANOVA ({side} side のみ): Experiment × Branch")
        print("=" * 70)

        df_side = df[df['side'] == side].copy()

        model = ols('intensity ~ C(experiment) * C(branch)', data=df_side).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print("\n【ANOVA Table】")
        print(anova_table)

        print("\n【有意性判定】")
        for idx in anova_table.index[:-1]:
            p_val = anova_table.loc[idx, 'PR(>F)']
            sig = get_significance(p_val)
            print(f"  {idx}: p = {p_val:.6f} {sig}")

        results[side] = anova_table

    return results


def run_ttest_branch_comparison(df):
    """
    各条件でのVentral branch vs Dorsal branch のt検定
    """
    print("\n" + "=" * 70)
    print("t検定: Ventral branch vs Dorsal branch (各条件)")
    print("=" * 70)

    results = []

    for experiment in ['control', 'KO']:
        for side in ['Ventral', 'Dorsal']:
            df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]

            ventral = df_subset[df_subset['branch'] == 'Ventral']['intensity']
            dorsal = df_subset[df_subset['branch'] == 'Dorsal']['intensity']

            t_stat, p_val = stats.ttest_ind(ventral, dorsal)
            sig = get_significance(p_val)

            diff = ventral.mean() - dorsal.mean()

            print(f"\n【{experiment} - {side} side】")
            print(f"  Ventral branch: mean = {ventral.mean():.2f}, n = {len(ventral)}")
            print(f"  Dorsal branch:  mean = {dorsal.mean():.2f}, n = {len(dorsal)}")
            print(f"  差 (V - D): {diff:.2f}")
            print(f"  t = {t_stat:.4f}, p = {p_val:.6f} {sig}")

            results.append({
                'experiment': experiment,
                'side': side,
                'ventral_mean': ventral.mean(),
                'dorsal_mean': dorsal.mean(),
                'diff': diff,
                't_stat': t_stat,
                'p_val': p_val,
                'sig': sig
            })

    return pd.DataFrame(results)


def print_descriptive_stats(df):
    """記述統計量を表示"""
    print("\n" + "=" * 70)
    print("記述統計量")
    print("=" * 70)

    grouped = df.groupby(['experiment', 'side', 'branch'])['intensity']
    stats_df = grouped.agg(['count', 'mean', 'std', 'median']).round(3)
    print(stats_df)


def run_posthoc_tukey(df):
    """Tukey HSD事後検定"""
    print("\n" + "=" * 70)
    print("事後検定 (Tukey HSD): Experiment × Branch")
    print("=" * 70)

    df_copy = df.copy()
    df_copy['group'] = df_copy['experiment'] + '_' + df_copy['branch']

    tukey = pairwise_tukeyhsd(df_copy['intensity'], df_copy['group'], alpha=0.05)
    print(tukey)
    return tukey


def plot_interaction_exp_branch(df, anova_table):
    """Experiment × Branch の交互作用プロット"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 平均値と標準誤差を計算
    summary = df.groupby(['experiment', 'branch'])['intensity'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.reset_index()

    # --- 左図: Experiment を x 軸 ---
    ax1 = axes[0]
    colors = {'Ventral': '#1f77b4', 'Dorsal': '#ff7f0e'}
    for branch in ['Ventral', 'Dorsal']:
        data = summary[summary['branch'] == branch]
        ax1.errorbar(
            data['experiment'], data['mean'], yerr=data['se'],
            marker='o', markersize=10, capsize=5, capthick=2, linewidth=2.5,
            label=f'{branch} branch', color=colors[branch]
        )
    ax1.set_xlabel('Experiment', fontsize=14)
    ax1.set_ylabel('Intensity (Mean ± SE)', fontsize=14)
    ax1.set_title('Interaction Plot\n(X: Experiment, Lines: Branch)', fontsize=13)
    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)

    # --- 右図: Branch を x 軸 ---
    ax2 = axes[1]
    colors_exp = {'control': '#2ca02c', 'KO': '#d62728'}
    for experiment in ['control', 'KO']:
        data = summary[summary['experiment'] == experiment]
        ax2.errorbar(
            data['branch'], data['mean'], yerr=data['se'],
            marker='o', markersize=10, capsize=5, capthick=2, linewidth=2.5,
            label=experiment, color=colors_exp[experiment]
        )
    ax2.set_xlabel('Branch', fontsize=14)
    ax2.set_ylabel('Intensity (Mean ± SE)', fontsize=14)
    ax2.set_title('Interaction Plot\n(X: Branch, Lines: Experiment)', fontsize=13)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)

    # ANOVA結果
    p_interaction = anova_table.loc['C(experiment):C(branch)', 'PR(>F)']
    sig = get_significance(p_interaction)
    fig.suptitle(f'Two-way ANOVA: Experiment × Branch\nInteraction p = {p_interaction:.2e} {sig}',
                 fontsize=14, y=1.05)

    plt.tight_layout()
    plt.savefig('twoway_anova_interaction.png', dpi=150, bbox_inches='tight')
    print("\n交互作用プロットを 'twoway_anova_interaction.png' に保存しました。")
    plt.show()


def plot_boxplot_by_condition(df, ttest_results):
    """条件別の箱ひげ図（t検定結果付き）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    conditions = [
        ('control', 'Ventral', axes[0, 0]),
        ('control', 'Dorsal', axes[0, 1]),
        ('KO', 'Ventral', axes[1, 0]),
        ('KO', 'Dorsal', axes[1, 1]),
    ]

    for experiment, side, ax in conditions:
        df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]

        # 箱ひげ図
        data_v = df_subset[df_subset['branch'] == 'Ventral']['intensity']
        data_d = df_subset[df_subset['branch'] == 'Dorsal']['intensity']

        box = ax.boxplot([data_v, data_d], tick_labels=['Ventral\nbranch', 'Dorsal\nbranch'],
                         patch_artist=True, widths=0.6)

        # スタイル
        colors_box = ['#1f77b4', '#ff7f0e']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors_box[i])
            patch.set_alpha(0.7)

        # n数
        ax.text(1, data_v.max() + 2, f'n={len(data_v)}', ha='center', fontsize=11)
        ax.text(2, data_d.max() + 2, f'n={len(data_d)}', ha='center', fontsize=11)

        # t検定結果
        result = ttest_results[(ttest_results['experiment'] == experiment) &
                               (ttest_results['side'] == side)].iloc[0]
        sig = result['sig']

        # 有意差線
        y_max = max(data_v.max(), data_d.max()) + 5
        ax.plot([1, 1, 2, 2], [y_max, y_max + 2, y_max + 2, y_max], lw=1.5, c='black')
        ax.text(1.5, y_max + 2, sig, ha='center', fontsize=16, fontweight='bold')

        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title(f'{experiment} - {side} side\n(p = {result["p_val"]:.2e})', fontsize=13)
        ax.set_ylim(0, y_max + 15)
        ax.tick_params(labelsize=11)

    plt.suptitle('Ventral branch vs Dorsal branch (by condition)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('twoway_anova_boxplot_conditions.png', dpi=150, bbox_inches='tight')
    print("条件別箱ひげ図を 'twoway_anova_boxplot_conditions.png' に保存しました。")
    plt.show()


def plot_branch_difference(df, ttest_results):
    """Branch間の差（Ventral - Dorsal）を比較するプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 各条件でのbranch間差を計算
    diff_data = []
    for experiment in ['control', 'KO']:
        for side in ['Ventral', 'Dorsal']:
            df_subset = df[(df['experiment'] == experiment) & (df['side'] == side)]

            # 各サンプルペアの差を計算（同じインデックス同士）
            ventral = df_subset[df_subset['branch'] == 'Ventral']['intensity'].values
            dorsal = df_subset[df_subset['branch'] == 'Dorsal']['intensity'].values

            # サンプル数が同じ場合、ペアごとの差を計算
            min_len = min(len(ventral), len(dorsal))
            differences = ventral[:min_len] - dorsal[:min_len]

            for d in differences:
                diff_data.append({
                    'experiment': experiment,
                    'side': side,
                    'diff': d
                })

    df_diff = pd.DataFrame(diff_data)
    df_diff['group'] = df_diff['experiment'] + '\n' + df_diff['side'] + ' side'

    # 箱ひげ図
    order = ['control\nVentral side', 'control\nDorsal side', 'KO\nVentral side', 'KO\nDorsal side']
    colors = ['#2ca02c', '#2ca02c', '#d62728', '#d62728']

    box = ax.boxplot([df_diff[df_diff['group'] == g]['diff'] for g in order],
                     tick_labels=order, patch_artist=True, widths=0.6)

    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Intensity Difference\n(Ventral branch - Dorsal branch)', fontsize=13)
    ax.set_title('Branch Difference by Experiment and Side\n(Positive = Ventral > Dorsal)', fontsize=14)
    ax.tick_params(labelsize=11)

    # 平均値を表示
    for i, g in enumerate(order):
        mean_val = df_diff[df_diff['group'] == g]['diff'].mean()
        ax.text(i + 1, ax.get_ylim()[1] * 0.9, f'mean={mean_val:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('twoway_anova_branch_difference.png', dpi=150, bbox_inches='tight')
    print("Branch差の比較図を 'twoway_anova_branch_difference.png' に保存しました。")
    plt.show()


def plot_summary_by_side(df, anova_by_side, ttest_results):
    """Side別のサマリー図（メイン結果）"""
    fig = plt.figure(figsize=(14, 10))

    colors = {'Ventral': '#1f77b4', 'Dorsal': '#ff7f0e'}
    colors_exp = {'control': '#2ca02c', 'KO': '#d62728'}

    # --- 1. Ventral side 交互作用プロット (左上) ---
    ax1 = fig.add_subplot(2, 2, 1)
    df_v = df[df['side'] == 'Ventral']
    summary_v = df_v.groupby(['experiment', 'branch'])['intensity'].agg(['mean', 'std', 'count'])
    summary_v['se'] = summary_v['std'] / np.sqrt(summary_v['count'])
    summary_v = summary_v.reset_index()

    for branch in ['Ventral', 'Dorsal']:
        data = summary_v[summary_v['branch'] == branch]
        ax1.errorbar(
            data['experiment'], data['mean'], yerr=data['se'],
            marker='o', markersize=10, capsize=5, capthick=2, linewidth=2.5,
            label=f'{branch} branch', color=colors[branch]
        )

    p_int_v = anova_by_side['Ventral'].loc['C(experiment):C(branch)', 'PR(>F)']
    ax1.set_xlabel('Experiment', fontsize=12)
    ax1.set_ylabel('Intensity (Mean ± SE)', fontsize=12)
    ax1.set_title(f'Ventral side\nInteraction: p = {p_int_v:.2e} {get_significance(p_int_v)}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)

    # --- 2. Dorsal side 交互作用プロット (右上) ---
    ax2 = fig.add_subplot(2, 2, 2)
    df_d = df[df['side'] == 'Dorsal']
    summary_d = df_d.groupby(['experiment', 'branch'])['intensity'].agg(['mean', 'std', 'count'])
    summary_d['se'] = summary_d['std'] / np.sqrt(summary_d['count'])
    summary_d = summary_d.reset_index()

    for branch in ['Ventral', 'Dorsal']:
        data = summary_d[summary_d['branch'] == branch]
        ax2.errorbar(
            data['experiment'], data['mean'], yerr=data['se'],
            marker='o', markersize=10, capsize=5, capthick=2, linewidth=2.5,
            label=f'{branch} branch', color=colors[branch]
        )

    p_int_d = anova_by_side['Dorsal'].loc['C(experiment):C(branch)', 'PR(>F)']
    ax2.set_xlabel('Experiment', fontsize=12)
    ax2.set_ylabel('Intensity (Mean ± SE)', fontsize=12)
    ax2.set_title(f'Dorsal side\nInteraction: p = {p_int_d:.2e} {get_significance(p_int_d)}', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)

    # --- 3. Branch差の棒グラフ (左下) ---
    ax3 = fig.add_subplot(2, 2, 3)

    x_labels = []
    diffs = []
    sigs = []
    colors_bar = []

    for _, row in ttest_results.iterrows():
        x_labels.append(f"{row['experiment']}\n{row['side']}")
        diffs.append(abs(row['diff']))  # 絶対値で表示
        sigs.append(row['sig'])
        colors_bar.append(colors_exp[row['experiment']])

    bars = ax3.bar(x_labels, diffs, color=colors_bar, alpha=0.7, edgecolor='black')

    for i, (bar, sig) in enumerate(zip(bars, sigs)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 sig, ha='center', fontsize=14, fontweight='bold')

    ax3.set_ylabel('|V - D| branch difference', fontsize=12)
    ax3.set_title('Branch Intensity Difference (absolute)\nwith t-test significance', fontsize=13)
    ax3.tick_params(labelsize=10)

    # --- 4. 結果サマリー (右下) ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = "Two-way ANOVA Results (by side)\n"
    summary_text += "=" * 45 + "\n\n"

    summary_text += "Ventral side (Experiment x Branch):\n"
    summary_text += f"  Interaction: p = {p_int_v:.2e} {get_significance(p_int_v)}\n\n"

    summary_text += "Dorsal side (Experiment x Branch):\n"
    summary_text += f"  Interaction: p = {p_int_d:.2e} {get_significance(p_int_d)}\n\n"

    summary_text += "=" * 45 + "\n"
    summary_text += "t-test: V branch vs D branch\n"
    summary_text += "-" * 45 + "\n"

    for _, row in ttest_results.iterrows():
        summary_text += f"{row['experiment']:8} {row['side']:8}: "
        summary_text += f"diff={row['diff']:+.1f}, p={row['p_val']:.1e} {row['sig']}\n"

    summary_text += "\n" + "=" * 45 + "\n"
    summary_text += "Conclusion:\n"
    summary_text += "KO reduces V-D branch intensity difference\n"
    summary_text += f"(control: ****, KO: **~***)"

    ax4.text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
             transform=ax4.transAxes, verticalalignment='top')

    plt.suptitle('Two-way ANOVA: Effect of KO on Branch Intensity Difference',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('twoway_anova_summary.png', dpi=150, bbox_inches='tight')
    print("\nサマリー図を 'twoway_anova_summary.png' に保存しました。")
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

    # t検定: 各条件でのVentral vs Dorsal branch
    ttest_results = run_ttest_branch_comparison(df)

    # Two-way ANOVA: Experiment × Branch（全データ）
    anova_table = run_twoway_anova_exp_branch(df)

    # Side別のTwo-way ANOVA（メイン解析）
    anova_by_side = run_twoway_anova_by_side(df)

    # 事後検定
    tukey = run_posthoc_tukey(df)

    # --- 図の生成 ---
    print("\n" + "=" * 70)
    print("図の生成")
    print("=" * 70)

    # 交互作用プロット（全体）
    plot_interaction_exp_branch(df, anova_table)

    # 条件別箱ひげ図
    plot_boxplot_by_condition(df, ttest_results)

    # Branch差の比較
    plot_branch_difference(df, ttest_results)

    # サマリー図（Side別、メイン結果）
    plot_summary_by_side(df, anova_by_side, ttest_results)
