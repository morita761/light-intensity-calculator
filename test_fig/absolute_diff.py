import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 全データの読み込み
cv = pd.read_csv('control_v.csv', sep=' ')
cd = pd.read_csv('control_d.csv', sep=' ')
kv = pd.read_csv('cas9_v.csv', sep=' ')
kd = pd.read_csv('cas9_d.csv', sep=' ')

# 2. 各サンプルの絶対値差を計算
c_all_diff = pd.concat([
    (cv['left_intensity'] - cv['right_intensity']).abs(),
    (cd['left_intensity'] - cd['right_intensity']).abs()
])
k_all_diff = pd.concat([
    (kv['left_intensity'] - kv['right_intensity']).abs(),
    (kd['left_intensity'] - kd['right_intensity']).abs()
])

# 3. まとめた状態でt検定
t_stat, p_val = stats.ttest_ind(c_all_diff, k_all_diff)

print(f"Pooled T-test P-value: {p_val:.2e}")
print(f"Control Mean: {c_all_diff.mean():.2f}, Cas9 Mean: {k_all_diff.mean():.2f}")

# 4. グラフ作成
plot_data = pd.DataFrame({
    'Genotype': ['Control'] * len(c_all_diff) + ['Cas9'] * len(k_all_diff),
    'Polarity (Abs Diff)': pd.concat([c_all_diff, k_all_diff])
})

plt.figure(figsize=(6, 6))
sns.barplot(data=plot_data, x='Genotype', y='Polarity (Abs Diff)', capsize=.1, palette='Set2')
plt.title(f'M1 Layer Overall: Polarity Attenuation\n(p = {p_val:.2e})')
plt.ylabel('Mean Absolute Difference (|L - R|)')
plt.savefig('pooled_comparison.png')
plt.show()