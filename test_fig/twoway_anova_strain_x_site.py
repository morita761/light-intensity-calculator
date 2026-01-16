import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# 3. 相互作用プロット (Interaction Plot) の作成
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

# 4. 輝度差（絶対値）の比較グラフ作成
# 各馬蹄形ごとの |Left - Right| を計算
def get_diff_df(df_ctrl, df_cas9):
    ctrl_diff = (df_ctrl['left_intensity'] - df_ctrl['right_intensity']).abs().to_frame(name='diff')
    ctrl_diff['genotype'] = 'Control'
    
    cas9_diff = (df_cas9['left_intensity'] - df_cas9['right_intensity']).abs().to_frame(name='diff')
    cas9_diff['genotype'] = 'Cas9'
    
    return pd.concat([ctrl_diff, cas9_diff])

v_diff_df = get_diff_df(control_v, cas9_v)
d_diff_df = get_diff_df(control_d, cas9_d)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

# Ventral 輝度差比較
sns.barplot(data=v_diff_df, x='genotype', y='diff', ax=axes2[0], palette='muted')
axes2[0].set_title('Ventral Side: Absolute Diff ($|L - R|$)')
axes2[0].set_ylabel('Mean Intensity Difference')

# Dorsal 輝度差比較
sns.barplot(data=d_diff_df, x='genotype', y='diff', ax=axes2[1], palette='muted')
axes2[1].set_title('Dorsal Side: Absolute Diff ($|L - R|$)')
axes2[1].set_ylabel('Mean Intensity Difference')

plt.tight_layout()
plt.savefig('difference_comparison.png')

plt.show()