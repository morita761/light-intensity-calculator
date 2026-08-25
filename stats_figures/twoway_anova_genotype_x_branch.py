import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. データの読み込み (スペース区切り)
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
cas9_v = pd.read_csv('cas9_v.csv', sep=' ')
cas9_d = pd.read_csv('cas9_d.csv', sep=' ')

# 2. データの整形用関数
def prepare_data(df_ctrl, df_cas9, side_label):
    # Controlデータの整形
    ctrl_melt = df_ctrl.melt(var_name='branch', value_name='intensity')
    ctrl_melt['genotype'] = 'Control'
    
    # Cas9(KO)データの整形
    cas9_melt = df_cas9.melt(var_name='branch', value_name='intensity')
    cas9_melt['genotype'] = 'Cas9'
    
    combined = pd.concat([ctrl_melt, cas9_melt])
    combined['side'] = side_label
    return combined

# Ventral側とDorsal側のデータを準備
data_v = prepare_data(control_v, cas9_v, 'Ventral')
data_d = prepare_data(control_d, cas9_d, 'Dorsal')

# 3. Two-way ANOVA の実行
# 要因: genotype (Control/Cas9) × branch (left/right)
print("=" * 60)
print("Two-way ANOVA Results")
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
print("- C(genotype): 系統間（Control vs Cas9）の主効果")
print("- C(branch): 部位間（Left vs Right）の主効果")
print("- C(genotype):C(branch): 交互作用（系統によって左右差が異なるか）")
print("=" * 60)

# 4. 相互作用プロット (Interaction Plot) の作成
# 系統によって部位間の傾き（差）がどう変わるかを可視化します
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ventral Side Plot
sns.pointplot(data=data_v, x='branch', y='intensity', hue='genotype', ax=axes[0], capsize=.1)
axes[0].set_title('Interaction Plot: Ventral Side')
axes[0].set_ylabel('GFP Intensity')
axes[0].set_xlabel('Branch')

# Dorsal Side Plot
sns.pointplot(data=data_d, x='branch', y='intensity', hue='genotype', ax=axes[1], capsize=.1)
axes[1].set_title('Interaction Plot: Dorsal Side')
axes[1].set_ylabel('GFP Intensity')
axes[1].set_xlabel('Branch')

plt.tight_layout()
plt.savefig('interaction_plots.png')
plt.show()