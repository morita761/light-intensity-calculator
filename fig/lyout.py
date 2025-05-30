import pandas as pd
import matplotlib.pyplot as plt

# サンプルデータ
data = {
    'left_intensity': [140.0, 150.0, 130.0, 160.0, 145.0],
    'right_intensity': [134.9, 120.0, 125.0, 140.0, 130.0]
}
df = pd.DataFrame(data)
plot_data = [df['left_intensity'], df['right_intensity']]
labels = ['Left', 'Right']

# タイトルを変数として定義
plot_title = "Comparison of Left and Right Intensity"

# 図の作成
fig, ax = plt.subplots(figsize=(6, 5))

# 箱ひげ図の描画（箱の横幅はwidthで調整）
box = ax.boxplot(plot_data, labels=labels, patch_artist=True, widths=0.6)

# 箱のスタイル設定（左：斜線、右：白色）
for i, patch in enumerate(box['boxes']):
    patch.set_facecolor("white")
    patch.set_edgecolor("black")
    patch.set_linewidth(1.5)
    if i == 0:  # Left box
        patch.set_hatch('///')  # 黒の斜線

# その他の要素のスタイル
for whisker in box['whiskers']:
    whisker.set_color('black')
    whisker.set_linewidth(1.5)

for cap in box['caps']:
    cap.set_color('black')
    cap.set_linewidth(1.5)

for median in box['medians']:
    median.set_color('red')
    median.set_linewidth(2)

for flier in box['fliers']:
    flier.set(marker='o', color='orange', markersize=6)

# 軸の調整
ax.set_ylim(0, 250)
ax.set_ylabel("Intensity", fontsize=14)
ax.set_xlabel("", fontsize=14)
ax.tick_params(labelsize=12)

# データ数を最大値の上に表示
for i, d in enumerate(plot_data):
    max_val = max(d)
    count = len(d)
    ax.text(i + 1, max_val + 5, f'n={count}', ha='center', va='bottom', fontsize=12)

# タイトルを図の下中央に表示
fig.text(0.5, 0.02, plot_title, ha='center', va='center', fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # 下部のタイトル表示スペースを確保
plt.show()
