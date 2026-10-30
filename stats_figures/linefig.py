"""複数CSVの列ごとに 最小〜最大の縦線 + 平均(×) + n数 を描く

用途: 列 = カテゴリ（例: C1, BA, C2）の形式のCSVを複数並べて、
      値の範囲と平均をファイルごとに色分けして比較する汎用グラフ。
      --files を省略するとサンプルデータで描画する（見た目の確認用）。
入力: --files で指定したCSV（カンマ区切り、1行目が列名）
出力: output/linefig/linefig.png

Usage:
    python linefig.py
    python linefig.py --files a.csv b.csv c.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.figure import finish, make_parser, save_figure
from lib.settings import load_figure_settings

NAME = 'linefig'

SAMPLE_DATA = {
    'sample1': {'C1': [0.95, 0.77, 0.87], 'BA': [0.55, 0.67, 0.77], 'C2': [0.85, 0.97, 0.66]},
    'sample2': {'C1': [0.89, 0.92, 0.81], 'BA': [0.71, 0.65, 0.73], 'C2': [0.78, 0.88, 0.90]},
    'sample3': {'C1': [0.75, 0.85, 0.69], 'BA': [0.90, 0.82, 0.85], 'C2': [0.65, 0.72, 0.78]},
}


def main():
    parser = make_parser(__doc__, NAME)
    parser.add_argument('--files', nargs='+', help='プロットするCSV（省略時はサンプルデータ）')
    args = parser.parse_args()
    s = load_figure_settings(NAME)

    if args.files:
        datasets = {Path(f).stem: pd.read_csv(f) for f in args.files}
    else:
        datasets = {name: pd.DataFrame(d) for name, d in SAMPLE_DATA.items()}
    colors = s.list('colors', ['red', 'blue', 'green'])
    n_offset = s.float('n_offset', 0.05)

    fig, ax = plt.subplots(figsize=s.floats('figsize', (8, 6)))
    columns = []
    for i, (name, df) in enumerate(datasets.items()):
        color = colors[i % len(colors)]
        columns = df.columns
        x_positions = np.arange(len(columns)) + 1 + i * 0.2
        for x, col in zip(x_positions, columns):
            col_data = df[col].dropna()
            max_val, min_val = col_data.max(), col_data.min()
            ax.plot([x, x], [min_val, max_val], color=color, linewidth=1.5)
            ax.hlines([max_val, min_val], x - 0.05, x + 0.05, color=color, linewidth=2)
            ax.scatter(x, col_data.mean(), marker='x', color=color, s=100)
            ax.text(x, max_val + n_offset, f'n={len(col_data)}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(np.arange(len(columns)) + 1 + 0.2 * (len(datasets) - 1) / 2)
    ax.set_xticklabels(columns)
    s.apply(ax)
    ax.legend(handles=[plt.Line2D([0], [0], color=colors[i % len(colors)], lw=4, label=name)
                       for i, name in enumerate(datasets)])

    fig.tight_layout()
    save_figure(fig, NAME, args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
