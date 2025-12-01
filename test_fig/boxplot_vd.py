import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

labels = ['Ventral', 'Dorsal']

# p値をアスタリスクに変換する関数
def get_pvalue_asterisk(p):
    if p < 0.00001:
        return '*****'
    elif p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


# 1つの箱ひげ図を描く関数 ---------------------------------------------
def plot_box(ax, df, title):

    data = [df['left_intensity'], df['right_intensity']]

    # 箱ひげ図
    box = ax.boxplot(data, labels=['Ventral branch', 'Dorsal branch'], patch_artist=True, widths=0.6)

    # boxスタイル
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        if i == 0:  # 左
            patch.set_hatch('///')

    for part in ['whiskers', 'caps', 'medians']:
        for element in box[part]:
            element.set_color('black')
            element.set_linewidth(1.5)

    for flier in box['fliers']:
        flier.set(marker='o', color='black', markersize=6)

    # 軸
    ax.set_ylabel("Intensity", fontsize=14)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', labelsize=12)

    # n 表示
    for i, d in enumerate(data):
        ax.text(i + 1, max(d) + 2, f'n={len(d)}', ha='center', fontsize=11)

    # t検定
    t_stat, p_val = ttest_ind(data[0], data[1])
    asters = get_pvalue_asterisk(p_val)

    # 線とアスタリスク
    y_max = max(max(data[0]), max(data[1])) + 11
    ax.plot([1, 1, 2, 2], [y_max, y_max + 2, y_max + 2, y_max],
            lw=1.5, c='black')
    ax.text(1.5, y_max + 2, asters, ha='center', fontsize=16)

    # タイトル
    ax.set_title(title, fontsize=15)


# ----------------------------------------------------------------------

# CSVを2つ読み込み
df_v = pd.read_csv("v_normalize.csv", delim_whitespace=True,
                   names=["left_intensity", "right_intensity"], skiprows=1)

df_d = pd.read_csv("d_normalize.csv", delim_whitespace=True,
                   names=["left_intensity", "right_intensity"], skiprows=1)

# subplotで横に並べる
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

plot_box(axes[0], df_v, "Intensity in Ventral R8s")
plot_box(axes[1], df_d, "Intensity in Dorsal R8s")

plt.tight_layout()
plt.show()
