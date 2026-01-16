import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. データの読み込み
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
cas9_v = pd.read_csv('cas9_v.csv', sep=' ')
cas9_d = pd.read_csv('cas9_d.csv', sep=' ')

# 2. データの整形
def melt_data(df, genotype):
    melted = df.melt(var_name='branch', value_name='intensity')
    melted['genotype'] = genotype
    return melted

data_all = pd.concat([
    melt_data(control_v, 'Control'), melt_data(control_d, 'Control'),
    melt_data(cas9_v, 'Cas9'), melt_data(cas9_d, 'Cas9')
])

def get_diff(df, genotype):
    diff = (df['left_intensity'] - df['right_intensity']).abs().to_frame(name='abs_diff')
    diff['genotype'] = genotype
    return diff

diff_all = pd.concat([
    get_diff(control_v, 'Control'), get_diff(control_d, 'Control'),
    get_diff(cas9_v, 'Cas9'), get_diff(cas9_d, 'Cas9')
])

# 3. プロット作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# A. Interaction Plot (個別ドット + 平均線)
sns.stripplot(data=data_all, x='branch', y='intensity', hue='genotype',
              dodge=True, alpha=0.3, palette='Set2', ax=ax1, jitter=True, size=3)
sns.pointplot(data=data_all, x='branch', y='intensity', hue='genotype',
              capsize=.1, markers="D", linestyles="-", ax=ax1, errorbar='se', palette='dark:black')

# 凡例の重複整理
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[0:2], labels[0:2], title='Genotype')
ax1.set_title('Interaction Plot (Pooled)\nIndividual points + Mean')

# B. Absolute Difference Bar Plot (個別ドット + 棒グラフ)
sns.barplot(data=diff_all, x='genotype', y='abs_diff', capsize=.1,
            palette='Set2', ax=ax2, errorbar='se', alpha=0.6)
sns.stripplot(data=diff_all, x='genotype', y='abs_diff',
              alpha=0.5, color='black', ax=ax2, jitter=True, size=3)

# 統計
t_stat, p_val = stats.ttest_ind(
    diff_all[diff_all['genotype']=='Control']['abs_diff'],
    diff_all[diff_all['genotype']=='Cas9']['abs_diff']
)
ax2.set_title(f'Polarity Magnitude (|L - R|)\nT-test P-value: {p_val:.2e}')

plt.tight_layout()
plt.show()