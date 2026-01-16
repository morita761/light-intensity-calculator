import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

# 4. 事後検定 (Tukey HSD) - 系統間の多重比較
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# 5. 相互作用プロット (Interaction Plot) の作成
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

# 6. 箱ひげ図も追加
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
