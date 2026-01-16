import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

# ANOVA結果を表として図に描画する関数
def plot_anova_table(anova_result, title, filename=None):
    """ANOVA結果をテーブル形式で図として表示"""
    # データを整形
    df = anova_result.copy()
    df = df.reset_index()
    df.columns = ['Source', 'Sum of Sq', 'df', 'F', 'PR(>F)']

    # ラベルを分かりやすく変更
    label_map = {
        'C(genotype)': 'Genotype',
        'C(branch)': 'Branch (L/R)',
        'C(genotype):C(branch)': 'Genotype × Branch',
        'Residual': 'Residual'
    }
    df['Source'] = df['Source'].map(lambda x: label_map.get(x, x))

    # 数値をフォーマット
    df['Sum of Sq'] = df['Sum of Sq'].apply(lambda x: f'{x:.2f}')
    df['df'] = df['df'].apply(lambda x: f'{int(x)}')
    df['F'] = df['F'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else '-')
    df['PR(>F)'] = df['PR(>F)'].apply(lambda x: f'{x:.2e}' if pd.notna(x) else '-')

    # 有意性マーカーを追加
    def add_significance(p_str):
        if p_str == '-':
            return '-'
        try:
            p = float(p_str)
            if p < 0.0001:
                return f'{p_str} ****'
            elif p < 0.001:
                return f'{p_str} ***'
            elif p < 0.01:
                return f'{p_str} **'
            elif p < 0.05:
                return f'{p_str} *'
            else:
                return f'{p_str} n.s.'
        except:
            return p_str

    df['PR(>F)'] = df['PR(>F)'].apply(add_significance)

    # 図を作成
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')

    # テーブルを作成
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(df.columns)
    )

    # スタイル調整
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # ヘッダーの文字色を白に
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig

def plot_tukey_table(tukey_result, title, filename=None):
    """Tukey HSD結果をテーブル形式で図として表示"""
    # Tukey結果をDataFrameに変換
    df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                      columns=tukey_result._results_table.data[0])

    # 数値をフォーマット
    df['meandiff'] = df['meandiff'].apply(lambda x: f'{float(x):.3f}')
    df['p-adj'] = df['p-adj'].apply(lambda x: f'{float(x):.4f}')
    df['lower'] = df['lower'].apply(lambda x: f'{float(x):.3f}')
    df['upper'] = df['upper'].apply(lambda x: f'{float(x):.3f}')

    # 図を作成
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('tight')
    ax.axis('off')

    # テーブルを作成
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#70AD47'] * len(df.columns)
    )

    # スタイル調整
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # ヘッダーの文字色を白に
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig

# 1. データの読み込み (スペース区切り)
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
vang_rnai_v = pd.read_csv('vang_rnai_v.csv', sep=' ')
vang_rnai_d = pd.read_csv('vang_rnai_d.csv', sep=' ')
vang_cas9_v = pd.read_csv('loco_vang_cas9_v.csv', sep=' ')
vang_cas9_d = pd.read_csv('loco_vang_cas9_d.csv', sep=' ')

# 2. データの整形用関数
def prepare_data_three_groups(df_ctrl, df_vang_rnai, df_vang_cas9, side_label):
    # Controlデータの整形
    ctrl_melt = df_ctrl.melt(var_name='branch', value_name='intensity')
    ctrl_melt['genotype'] = 'Control'

    # vang RNAiデータの整形
    vang_rnai_melt = df_vang_rnai.melt(var_name='branch', value_name='intensity')
    vang_rnai_melt['genotype'] = 'vang RNAi'

    # vang Cas9データの整形
    vang_cas9_melt = df_vang_cas9.melt(var_name='branch', value_name='intensity')
    vang_cas9_melt['genotype'] = 'vang Cas9'

    combined = pd.concat([ctrl_melt, vang_rnai_melt, vang_cas9_melt])
    combined['side'] = side_label
    return combined

# Ventral側とDorsal側のデータを準備
data_v = prepare_data_three_groups(control_v, vang_rnai_v, vang_cas9_v, 'Ventral')
data_d = prepare_data_three_groups(control_d, vang_rnai_d, vang_cas9_d, 'Dorsal')

# 順序を指定
genotype_order = ['Control', 'vang RNAi', 'vang Cas9']

# 3. Two-way ANOVA の実行
# 要因: genotype (Control/vang RNAi/vang Cas9) × branch (left/right)
print("=" * 60)
print("Two-way ANOVA Results (3 Genotypes)")
print("=" * 60)

# Ventral側のTwo-way ANOVA
print("\n[Ventral Side]")
model_v = ols('intensity ~ C(genotype) * C(branch)', data=data_v).fit()
anova_v = sm.stats.anova_lm(model_v, typ=2)
print(anova_v)

# Dorsal側のTwo-way ANOVA
print("\n[Dorsal Side]")
model_d = ols('intensity ~ C(genotype) * C(branch)', data=data_d).fit()
anova_d = sm.stats.anova_lm(model_d, typ=2)
print(anova_d)

print("\n" + "=" * 60)
print("解釈:")
print("- C(genotype): 系統間（Control vs vang RNAi vs vang Cas9）の主効果")
print("- C(branch): 部位間（Left vs Right）の主効果")
print("- C(genotype):C(branch): 交互作用（系統によって左右差が異なるか）")
print("=" * 60)

# 4. ANOVA結果を図として保存
plot_anova_table(anova_v, 'Two-way ANOVA: Ventral Side', 'anova_table_ventral.png')
plot_anova_table(anova_d, 'Two-way ANOVA: Dorsal Side', 'anova_table_dorsal.png')

# 5. 事後検定 (Tukey HSD) - 系統間の多重比較
print("\n" + "=" * 60)
print("Post-hoc Test (Tukey HSD) - Genotype Comparison")
print("=" * 60)

# Ventral側の事後検定
print("\n[Ventral Side - Genotype Comparison]")
tukey_v = pairwise_tukeyhsd(data_v['intensity'], data_v['genotype'], alpha=0.05)
print(tukey_v)

# Dorsal側の事後検定
print("\n[Dorsal Side - Genotype Comparison]")
tukey_d = pairwise_tukeyhsd(data_d['intensity'], data_d['genotype'], alpha=0.05)
print(tukey_d)

# Tukey HSD結果を図として保存
plot_tukey_table(tukey_v, 'Tukey HSD: Ventral Side', 'tukey_table_ventral.png')
plot_tukey_table(tukey_d, 'Tukey HSD: Dorsal Side', 'tukey_table_dorsal.png')

# 7. 相互作用プロット (Interaction Plot) の作成
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ventral Side Plot
sns.pointplot(data=data_v, x='branch', y='intensity', hue='genotype',
              hue_order=genotype_order, ax=axes[0], capsize=.1)
axes[0].set_title('Interaction Plot: Ventral Side')
axes[0].set_ylabel('GFP Intensity')
axes[0].set_xlabel('Branch')
axes[0].legend(title='Genotype')

# Dorsal Side Plot
sns.pointplot(data=data_d, x='branch', y='intensity', hue='genotype',
              hue_order=genotype_order, ax=axes[1], capsize=.1)
axes[1].set_title('Interaction Plot: Dorsal Side')
axes[1].set_ylabel('GFP Intensity')
axes[1].set_xlabel('Branch')
axes[1].legend(title='Genotype')

plt.tight_layout()
plt.savefig('interaction_plots_three_groups.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 箱ひげ図も追加
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ventral Side Boxplot
sns.boxplot(data=data_v, x='genotype', y='intensity', hue='branch',
            order=genotype_order, ax=axes[0])
axes[0].set_title('Boxplot: Ventral Side')
axes[0].set_ylabel('GFP Intensity')
axes[0].set_xlabel('Genotype')

# Dorsal Side Boxplot
sns.boxplot(data=data_d, x='genotype', y='intensity', hue='branch',
            order=genotype_order, ax=axes[1])
axes[1].set_title('Boxplot: Dorsal Side')
axes[1].set_ylabel('GFP Intensity')
axes[1].set_xlabel('Genotype')

plt.tight_layout()
plt.savefig('boxplot_three_groups.png', dpi=300, bbox_inches='tight')
plt.show()
