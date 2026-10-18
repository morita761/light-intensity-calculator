"""複数スクリプトで使う作図部品"""
import pandas as pd
from scipy.stats import ttest_ind

from .data_io import COLUMNS
from .stats_utils import p_to_asterisks


def vd_boxplot(ax, df, s):
    """左右branch（left_intensity / right_intensity）の白黒箱ひげ図 + n数 + t検定ブラケット

    設定キー: branch_labels, n_fontsize, star_fontsize, bracket_offset, bracket_height
    返り値: (t統計量, p値)
    """
    data = [df[c].dropna() for c in COLUMNS]
    box = ax.boxplot(data, tick_labels=s.list('branch_labels', ['Ventral branch', 'Dorsal branch']),
                     patch_artist=True, widths=0.6)

    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        if i == 0:
            patch.set_hatch('///')  # 左は黒の斜線
    for part in ('whiskers', 'caps', 'medians'):
        for element in box[part]:
            element.set_color('black')
            element.set_linewidth(1.5)
    for flier in box['fliers']:
        flier.set(marker='o', color='black', markersize=6)

    # 各箱の最大値の上にn数
    for i, d in enumerate(data):
        if not d.empty:
            ax.text(i + 1, d.max() + 1, f'n={len(d)}', ha='center', va='bottom',
                    fontsize=s.float('n_fontsize', 11))

    # t検定（独立2群）
    t_stat, p_val = ttest_ind(*data)
    asterisks = p_to_asterisks(p_val, max_stars=s.int('max_stars', 5))

    max_val = max(d.max() for d in data if not d.empty)
    line_y = (max_val if not pd.isna(max_val) else 80) + s.float('bracket_offset', 8)
    height = s.float('bracket_height', 2)
    ax.plot([1, 1, 2, 2], [line_y, line_y + height, line_y + height, line_y], lw=1.5, c='black')
    ax.text(1.5, line_y + height / 2, asterisks, ha='center', va='bottom',
            fontsize=s.float('star_fontsize', 16))
    return t_stat, p_val
