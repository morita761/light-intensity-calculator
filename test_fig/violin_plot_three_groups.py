import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. データの読み込み (スペース区切り)
control_v = pd.read_csv('control_v.csv', sep=' ')
control_d = pd.read_csv('control_d.csv', sep=' ')
vang_rnai_v = pd.read_csv('vang_rnai_v.csv', sep=' ')
vang_rnai_d = pd.read_csv('vang_rnai_d.csv', sep=' ')
loco_vang_cas9_v = pd.read_csv('loco_vang_cas9_v.csv', sep=' ')
loco_vang_cas9_d = pd.read_csv('loco_vang_cas9_d.csv', sep=' ')

# 2. 絶対値の差を計算して結合する関数
def get_diff_df(df, genotype, side):
    diff = (df['left_intensity'] - df['right_intensity']).abs().to_frame(name='abs_diff')
    diff['genotype'] = genotype
    diff['side'] = side
    return diff

diff_all = pd.concat([
    get_diff_df(control_v, 'Control', 'Ventral'),
    get_diff_df(control_d, 'Control', 'Dorsal'),
    get_diff_df(vang_rnai_v, 'vang RNAi', 'Ventral'),
    get_diff_df(vang_rnai_d, 'vang RNAi', 'Dorsal'),
    get_diff_df(loco_vang_cas9_v, 'vang Cas9', 'Ventral'),
    get_diff_df(loco_vang_cas9_d, 'vang Cas9', 'Dorsal')
])

# 順序を指定
genotype_order = ['Control', 'vang RNAi', 'vang Cas9']

# 3. 統計計算とアスタリスクの判定
def get_asterisks(p_val):
    if p_val < 0.0001: return "****"
    elif p_val < 0.001: return "***"
    elif p_val < 0.01: return "**"
    elif p_val < 0.05: return "*"
    else: return "n.s."

# Control vs vang RNAi
control_data = diff_all[diff_all['genotype'] == 'Control']['abs_diff']
vang_rnai_data = diff_all[diff_all['genotype'] == 'vang RNAi']['abs_diff']
vang_cas9_data = diff_all[diff_all['genotype'] == 'vang Cas9']['abs_diff']

_, p_val_rnai = stats.ttest_ind(control_data, vang_rnai_data)
_, p_val_cas9 = stats.ttest_ind(control_data, vang_cas9_data)

asterisks_rnai = get_asterisks(p_val_rnai)
asterisks_cas9 = get_asterisks(p_val_cas9)

print(f"Control vs vang RNAi: P = {p_val_rnai:.2e} ({asterisks_rnai})")
print(f"Control vs vang Cas9: P = {p_val_cas9:.2e} ({asterisks_cas9})")

# 4. split=True を用いたバイオリンプロットの作成
plt.figure(figsize=(12, 7))

# hue='side' と split=True を組み合わせる
ax = sns.violinplot(data=diff_all, x='genotype', y='abs_diff', hue='side',
               split=True, inner='quartile', palette='Set2', alpha=0.7,
               order=genotype_order)

# 3系統 × 2サイド(split) = 6つの半バイオリンがあり、各3本ずつ線がある
for i, line in enumerate(ax.lines):
    if i % 3 == 1: # 0,1,2の「1」が中央値
        line.set_color('red')      # 赤色
        line.set_linewidth(2.5)    # 太く
        line.set_linestyle('-')    # 実線に変更
        line.set_alpha(1.0)

# 個別のドットを追加 (dodge=Trueにすることで左右に分かれたバイオリンに合わせる)
sns.stripplot(data=diff_all, x='genotype', y='abs_diff', hue='side',
              dodge=True, color='black', alpha=0.2, jitter=True, size=2, legend=False,
              order=genotype_order)

# 5. 線とアスタリスクの描画 (動的な位置決定)
max_val = diff_all['abs_diff'].max()

# Control vs vang RNAi (x=0 と x=1)
line_y1 = max_val + 5
star_y1 = line_y1 + 1
ax.plot([0, 0, 1, 1], [line_y1, line_y1 + 1, line_y1 + 1, line_y1], lw=1.5, c='black')
ax.text(0.5, star_y1, asterisks_rnai, ha='center', va='bottom', fontsize=14)

# Control vs vang Cas9 (x=0 と x=2)
line_y2 = max_val + 15
star_y2 = line_y2 + 1
ax.plot([0, 0, 2, 2], [line_y2, line_y2 + 1, line_y2 + 1, line_y2], lw=1.5, c='black')
ax.text(1.0, star_y2, asterisks_cas9, ha='center', va='bottom', fontsize=14)

plt.title(f'Polarity Magnitude (|L - R|) Comparison\nSplit Violin Plot (Ventral vs Dorsal)')
plt.ylabel('Absolute Intensity Difference')
plt.xlabel('Genotype')
plt.ylim(0, star_y2 + 8) # 上部に余白を作る
plt.tight_layout()
plt.savefig('violin_plot_three_groups.png', dpi=300, bbox_inches='tight')
plt.show()
