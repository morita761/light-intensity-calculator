import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# # 箱ひげ図を描くためのデータを整形
labels = ['Ventral branch', 'Dorsal branch']
# アスタリスクに変換する関数
def get_pvalue_asterisk(p):
    """p値を統計的有意性を示すアスタリスクに変換する関数"""
    if p < 0.00001:  # 0.00001未満
        print(p)
        return '*****'
    elif p < 0.0001: # 0.0001未満
        print(p)
        return '****'
    elif p < 0.001: # 0.001未満
        print(p)
        return '***'
    elif p < 0.01: # 0.01未満
        print(p)
        return '**'
    elif p < 0.05: # 0.05未満
        print(p)
        return '*'
    else:
        return 'n.s.'

def process_and_plot(filename, plot_title):
    """
    指定されたCSVファイルを読み込み、t検定を行い、箱ひげ図をプロットする関数。
    
    Args:
        filename (str): 読み込むCSVファイル名。
        plot_title (str): プロットの下部に表示するタイトル。
    """
    
    print(f"--- {plot_title}の処理を開始 ---")

    # CSV読み込み
    # カラム名の確認（スペースを含むカラム名があるので注意）
    try:
        df = pd.read_csv(filename, delim_whitespace=True, names=["left_intensity", "right_intensity"], skiprows=1)
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return

    # 箱ひげ図用のデータ
    data = [df['left_intensity'], df['right_intensity']]

    # t検定（独立2群のt検定）
    t_stat, p_val = ttest_ind(df['left_intensity'], df['right_intensity'])
    print(f"t検定結果: t-statistic = {t_stat:.4f}, p-value = {p_val:.5f}")
    asterisks = get_pvalue_asterisk(p_val)
    print(f"統計的有意性: {asterisks}")


    # プロット
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
    ax.set_ylabel("Normalized Fluorescence Intensity", fontsize=16)
    ax.set_ylim(0, 100)  # y軸を0〜100に設定
    
    ax.tick_params(axis='both', labelsize=14)

    # 各箱の最大値の上にデータの個数を表示
    for i, d in enumerate(data):
        # 最大値の計算にNaNを無視するため、dropna()を使用
        max_val = d.dropna().max()
        count = len(d.dropna()) # 欠損値を除いた個数を表示
        # max_valがNaNの場合に備えてチェック
        if not pd.isna(max_val):
            ax.text(i + 1, max_val + 1, f'n={count}', ha='center', va='bottom', fontsize=13)

    # 線とアスタリスクの描画
    # データ全体の最大値を取得 (NaNを除外)
    max_data_val = max(d.dropna().max() for d in data if not d.empty and not d.dropna().empty)
    
    # 描画位置を動的に決定
    y_max = max_data_val + 5 if not pd.isna(max_data_val) else 80 
    line_y = y_max + 3
    star_y = line_y + 1

    # 横線
    ax.plot([1, 1, 2, 2], [line_y, line_y + 2, line_y + 2, line_y], lw=1.5, c='black')

    # アスタリスク
    ax.text(1.5, star_y, asterisks, ha='center', va='bottom', fontsize=18)

    # タイトルを図の下中央に表示
    fig.text(0.5, 0.02, plot_title, ha='center', va='center', fontsize=17)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 下部のタイトル表示スペースを確保
    plt.show()
    print(f"--- {plot_title}の処理を完了 ---")
    print("-" * 40)


## 実行部分

# 1. "v_normalize.csv" のプロット
process_and_plot("control_v.csv", "Intensity in Ventral R8s")

# 2. "d_normalize.csv" のプロット
process_and_plot("control_d.csv", "Intensity in Dorsal R8s")


# 1. "v_normalize.csv" のプロット
process_and_plot("cas9_v.csv", "Intensity in Ventral R8s")

# 2. "d_normalize.csv" のプロット
process_and_plot("cas9_d.csv", "Intensity in Dorsal R8s")