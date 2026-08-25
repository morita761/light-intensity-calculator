import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# サンプルCSVファイルの作成（実際のファイルに置き換えてください）
data1 = {
    'C1': [0.95, 0.77, 0.87],
    'BA': [0.55, 0.67, 0.77],
    'C2': [0.85, 0.97, 0.66]
}
df1 = pd.DataFrame(data1)
df1.to_csv('sample1.csv', index=False)

data2 = {
    'C1': [0.89, 0.92, 0.81],
    'BA': [0.71, 0.65, 0.73],
    'C2': [0.78, 0.88, 0.90]
}
df2 = pd.DataFrame(data2)
df2.to_csv('sample2.csv', index=False)

data3 = {
    'C1': [0.75, 0.85, 0.69],
    'BA': [0.90, 0.82, 0.85],
    'C2': [0.65, 0.72, 0.78]
}
df3 = pd.DataFrame(data3)
df3.to_csv('sample3.csv', index=False)

# プロットするCSVファイルと色の定義
files = ['sample1.csv', 'sample2.csv', 'sample3.csv']
colors = ['red', 'blue', 'green']

# 図の作成
fig, ax = plt.subplots(figsize=(8, 6))

# 各ファイルに対してループでプロットを実行
for i, file in enumerate(files):
    df = pd.read_csv(file)
    columns = df.columns
    
    # 棒グラフの位置を計算
    x_positions = np.arange(len(columns)) + 1 + i * 0.2
    
    for j, col in enumerate(columns):
        col_data = df[col]
        
        # 最大値、最小値、平均値、データ数を計算
        max_val = col_data.max()
        min_val = col_data.min()
        mean_val = col_data.mean()
        count = len(col_data)
        
        # 線グラフのプロット
        plt.plot([x_positions[j], x_positions[j]], [min_val, max_val], color=colors[i], linewidth=1.5)
        
        # 最大値と最小値を横棒でプロット
        plt.hlines(max_val, x_positions[j] - 0.05, x_positions[j] + 0.05, color=colors[i], linewidth=2)
        plt.hlines(min_val, x_positions[j] - 0.05, x_positions[j] + 0.05, color=colors[i], linewidth=2)
        
        # 平均値をバツ印でプロット
        plt.scatter(x_positions[j], mean_val, marker='x', color=colors[i], s=100)
        
        # データ数を最大値の上に表示
        ax.text(x_positions[j], max_val + 0.05, f'n={count}', ha='center', va='bottom', fontsize=10)

# 軸の調整とラベルの設定
ax.set_ylim(0, 1.2)
ax.set_xticks(np.arange(len(columns)) + 1 + 0.2)
ax.set_xticklabels(columns)
ax.set_ylabel("Intensity", fontsize=14)
ax.set_xlabel("", fontsize=14)
ax.tick_params(labelsize=12)

# 凡例の作成
legend_handles = [plt.Line2D([0], [0], color=c, lw=4, label=f.replace('.csv', '')) for f, c in zip(files, colors)]
ax.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig('multi_colored_plot.png')
print("Plot saved as multi_colored_plot.png")