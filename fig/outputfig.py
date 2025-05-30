import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# pip install scipy

# # CSV読み込み
# df = pd.read_csv('data.csv')  # ファイル名を適宜変更してください

# # 箱ひげ図を描くためのデータを整形
# data = [df['left_intensity'], df['right_intensity']]
labels = ['Ventral', 'Dorsal']

# 空白・タブ区切りのファイルを読み込む（複数空白やタブもOK）
df = pd.read_csv("data.txt", delim_whitespace=True)

# カラム名の確認（スペースを含むカラム名があるので注意）
print(df.columns)
df = pd.read_csv("data.txt", delim_whitespace=True, names=["left_intensity", "right_intensity"], skiprows=1)
data = [df['left_intensity'], df['right_intensity']]

# タイトルを変数として定義
# plot_title = "Intensity in Ventral R8s (Vg Knowout Cas9P2)"
plot_title = "Intensity in Dorsal R8s (Vg Knowout Cas9P2)"
# plot_title = "Intensity in Ventral R8s (Control)"

# t検定（独立2群のt検定）
t_stat, p_val = ttest_ind(df['left_intensity'], df['right_intensity'])

# アスタリスクに変換する関数
def get_pvalue_asterisk(p):
    if p < 0.001:
        print(p)
        return '***'
    elif p < 0.01:
        print(p)
        return '**'
    elif p < 0.05:
        print(p)
        return '*'
    else:
        print(p)
        return 'n.s.'


# プロット
# fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(6, 5))

# 箱ひげ図の作成
box = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

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
    median.set_color('black')
    median.set_linewidth(2)

for flier in box['fliers']:
    flier.set(marker='o', color='black', markersize=6)

# 軸の設定
ax.set_ylabel("Intensity", fontsize=16)
ax.set_ylim(0, 250)  # y軸を0〜250に設定（必要なら200に）
# x軸ラベル（今回は空欄にしているけど入れるならここで）
# ax.set_xlabel("Condition", fontsize=16)

ax.tick_params(axis='both', labelsize=14)

# 各箱の最大値の上にデータの個数を表示
for i, d in enumerate(data):
    max_val = max(d)
    count = len(d)
    ax.text(i + 1, max_val + 5, f'n={count}', ha='center', va='bottom', fontsize=13)

# 線とアスタリスクの描画
y_max = max(max(data[0]), max(data[1])) + 15
line_y = y_max + 5
star_y = line_y + 3

# 横線
ax.plot([1, 1, 2, 2], [line_y, line_y + 2, line_y + 2, line_y], lw=1.5, c='black')

# アスタリスク
asterisks = get_pvalue_asterisk(p_val)
ax.text(1.5, star_y, asterisks, ha='center', va='bottom', fontsize=18)

# タイトルを図の下中央に表示
fig.text(0.5, 0.02, plot_title, ha='center', va='center', fontsize=17)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # 下部のタイトル表示スペースを確保
plt.show()
