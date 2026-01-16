import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. データの読み込み (スペース区切り)
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
cas9_v = pd.read_csv('cas9_v.csv', sep=' ')
cas9_d = pd.read_csv('cas9_d.csv', sep=' ')

# 2. 絶対値の差を計算して結合する関数
def get_diff_df(df, genotype, side):
    diff = (df['left_intensity'] - df['right_intensity']).abs().to_frame(name='abs_diff')
    diff['genotype'] = genotype
    diff['side'] = side
    return diff

diff_all = pd.concat([
    get_diff_df(control_v, 'Control', 'Ventral'),
    get_diff_df(control_d, 'Control', 'Dorsal'),
    get_diff_df(cas9_v, 'Cas9', 'Ventral'),
    get_diff_df(cas9_d, 'Cas9', 'Dorsal')
])

# 3. split=True を用いたバイオリンプロットの作成
plt.figure(figsize=(10, 7))

# hue='side' と split=True を組み合わせる
ax = sns.violinplot(data=diff_all, x='genotype', y='abs_diff', hue='side',
               split=True, inner='quartile', palette='Set2', alpha=0.7)

# 2系統 × 2サイド(split) = 4つの半バイオリンがあり、各3本ずつ線がある
for i, line in enumerate(ax.lines):
    if i % 3 == 1: # 0,1,2の「1」が中央値
        line.set_color('red')      # 赤色
        line.set_linewidth(2.5)    # 太く
        line.set_linestyle('-')    # 実線に変更
        line.set_alpha(1.0)

# 個別のドットを追加 (dodge=Trueにすることで左右に分かれたバイオリンに合わせる)
sns.stripplot(data=diff_all, x='genotype', y='abs_diff', hue='side',
              dodge=True, color='black', alpha=0.2, jitter=True, size=2, legend=False)
# 統計数値の計算
t_stat, p_val = stats.ttest_ind(
    diff_all[diff_all['genotype']=='Control']['abs_diff'],
    diff_all[diff_all['genotype']=='Cas9']['abs_diff']
)

plt.title(f'Polarity Magnitude (|L - R|) Comparison\nSplit Violin Plot (Ventral vs Dorsal)(P = {p_val:.2e})')
plt.ylabel('Absolute Intensity Difference')
plt.xlabel('Genotype')

plt.tight_layout()
plt.show()