import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'IPAGothic', 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
# japanize-matplotlibがインストールされている場合は使用
try:
    import japanize_matplotlib
except ImportError:
    pass

# P値を科学的表記でフォーマットする関数
def format_p_value_scientific(p):
    """P値を科学的表記でフォーマット（例: 6.27e-8）"""
    if pd.isna(p):
        return '-'

    # 有意性マーカーを決定
    if p < 0.0001:
        sig = '****'
    elif p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    # P値のフォーマット
    if p < 0.0001:
        # 科学的表記: 6.27e-8 の形式（文字化けしない）
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10 ** exponent)
        return f'{mantissa:.2f}e{exponent} {sig}'
    else:
        return f'{p:.4f} {sig}'

# 2つのANOVA結果を1つの表にまとめて描画する関数
def plot_combined_anova_table(anova_v, anova_d, title, filename=None):
    """VentralとDorsalのANOVA結果を1つの表にまとめて表示"""

    def process_anova(anova_result, target_label):
        df = anova_result.copy()
        df = df.reset_index()
        df.columns = ['要因', '平方和', 'df', 'F', 'P値']

        # 日本語ラベルに変更
        label_map = {
            'C(genotype)': '系統 (Genotype)',
            'C(branch)': '部位 (Left/Right)',
            'C(genotype):C(branch)': '交互作用 (系統×部位)',
            'Residual': '残差'
        }
        df['要因'] = df['要因'].map(lambda x: label_map.get(x, x))

        # 残差行を除外
        df = df[df['要因'] != '残差']

        # 解析対象列を追加
        df['解析対象'] = target_label

        return df

    # VentralとDorsalのデータを処理
    df_v = process_anova(anova_v, 'Ventral')
    df_d = process_anova(anova_d, 'Dorsal')

    # 結合
    df_combined = pd.concat([df_v, df_d], ignore_index=True)

    # 表示用に整形
    df_display = df_combined[['解析対象', '要因', 'df', 'F', 'P値']].copy()

    # 数値をフォーマット
    df_display['df'] = df_display['df'].apply(lambda x: f'{int(x)}')
    df_display['F'] = df_display['F'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    df_display['P値'] = df_combined['P値'].apply(format_p_value_scientific)

    # 図を作成
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # テーブルを作成
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(df_display.columns)
    )

    # スタイル調整
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # ヘッダーの文字色を白に
    for i in range(len(df_display.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Ventral/Dorsalで背景色を分ける
    for i in range(len(df_display)):
        if df_display.iloc[i]['解析対象'] == 'Ventral':
            for j in range(len(df_display.columns)):
                table[(i+1, j)].set_facecolor('#E6F3FF')
        else:
            for j in range(len(df_display.columns)):
                table[(i+1, j)].set_facecolor('#FFF2E6')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig

# 2つのTukey HSD結果を1つの表にまとめて描画する関数
def plot_combined_tukey_table(tukey_v, tukey_d, title, filename=None):
    """VentralとDorsalのTukey HSD結果を1つの表にまとめて表示（日本語）"""

    def process_tukey(tukey_result, target_label):
        df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                          columns=tukey_result._results_table.data[0])
        df['解析対象'] = target_label
        return df

    # VentralとDorsalのデータを処理
    df_v = process_tukey(tukey_v, 'Ventral')
    df_d = process_tukey(tukey_d, 'Dorsal')

    # 結合
    df_combined = pd.concat([df_v, df_d], ignore_index=True)

    # P値のフォーマット（有意性マーカー付き）
    def format_tukey_p(p):
        p_val = float(p)
        # P値が0または非常に小さい場合の処理
        if p_val <= 0 or np.isnan(p_val) or np.isinf(p_val):
            return '< 1e-15 ****'
        if p_val < 0.0001:
            sig = '****'
            exponent = int(np.floor(np.log10(p_val)))
            mantissa = p_val / (10 ** exponent)
            return f'{mantissa:.2f}e{exponent} {sig}'
        elif p_val < 0.001:
            return f'{p_val:.4f} ***'
        elif p_val < 0.01:
            return f'{p_val:.4f} **'
        elif p_val < 0.05:
            return f'{p_val:.4f} *'
        else:
            return f'{p_val:.4f} n.s.'

    # 表示用に整形
    df_display = pd.DataFrame()
    df_display['解析対象'] = df_combined['解析対象']
    df_display['比較1'] = df_combined['group1']
    df_display['比較2'] = df_combined['group2']
    df_display['平均差'] = df_combined['meandiff'].apply(lambda x: f'{float(x):.3f}')
    df_display['調整済P値'] = df_combined['p-adj'].apply(format_tukey_p)
    df_display['有意'] = df_combined['reject'].apply(lambda x: 'Yes' if x else 'No')

    # 図を作成
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('tight')
    ax.axis('off')

    # テーブルを作成
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        colColours=['#70AD47'] * len(df_display.columns)
    )

    # スタイル調整
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # ヘッダーの文字色を白に
    for i in range(len(df_display.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Ventral/Dorsalで背景色を分ける
    for i in range(len(df_display)):
        if df_display.iloc[i]['解析対象'] == 'Ventral':
            for j in range(len(df_display.columns)):
                table[(i+1, j)].set_facecolor('#E6FFE6')
        else:
            for j in range(len(df_display.columns)):
                table[(i+1, j)].set_facecolor('#FFFFE6')

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

# 4. ANOVA結果を図として保存（VentralとDorsalを1つの表に統合）
plot_combined_anova_table(anova_v, anova_d, 'Two-way ANOVA Results', 'anova_table_combined.png')

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

# Tukey HSD結果を図として保存（VentralとDorsalを1つの表に統合）
plot_combined_tukey_table(tukey_v, tukey_d, 'Tukey HSD 多重比較結果', 'tukey_table_combined.png')

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
