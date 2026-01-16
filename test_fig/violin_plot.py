import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. データの読み込み (スペース区切りであることを想定)
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
cas9_v = pd.read_csv('cas9_v.csv', sep=' ')
cas9_d = pd.read_csv('cas9_d.csv', sep=' ')

# 2. 各サンプルの絶対値差を計算して結合
def get_diff(df, genotype):
    diff = (df['left_intensity'] - df['right_intensity']).abs().to_frame(name='abs_diff')
    diff['genotype'] = genotype
    return diff

diff_all = pd.concat([
    get_diff(control_v, 'Control'), get_diff(control_d, 'Control'),
    get_diff(cas9_v, 'Cas9'), get_diff(cas9_d, 'Cas9')
])

# 3. プロットの作成
plt.figure(figsize=(8, 6))

# バイオリンプロット (膨らみで密度を表現)
# inner=Noneにすることで、中の箱ひげ図を消してドットを見やすくしています
sns.violinplot(data=diff_all, x='genotype', y='abs_diff', 
               palette='Set2', inner='quartile', alpha=0.5)

# 個別の点を重ねる
sns.stripplot(data=diff_all, x='genotype', y='abs_diff', 
              color='black', alpha=0.4, jitter=True, size=3)

# 統計数値の計算
t_stat, p_val = stats.ttest_ind(
    diff_all[diff_all['genotype']=='Control']['abs_diff'],
    diff_all[diff_all['genotype']=='Cas9']['abs_diff']
)

# グラフの装飾
plt.title(f'Polarity Magnitude (|L - R|) Comparison\nViolin Plot with Individual Points (P = {p_val:.2e})')
plt.ylabel('Absolute Intensity Difference')
plt.xlabel('Genotype')

plt.tight_layout()
plt.show()