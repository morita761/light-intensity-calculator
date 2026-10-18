"""向き解析の作図部品"""
import math

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def bin_edges(bin_width_deg):
    """-180°〜180° を bin_width_deg ごとに区切ったビン境界（ラジアン）"""
    return np.linspace(-np.pi, np.pi, int(360 / bin_width_deg) + 1, endpoint=False)


def polar_hist(ax, angles_deg, s, color, label=None, normalize=True):
    """極座標ヒストグラムを描き、棒の高さの配列を返す（データなしなら空配列）"""
    if not angles_deg:
        return np.array([])
    counts, edges = np.histogram(np.radians(angles_deg), bins=bin_edges(s.float('bin_width_deg', 10.0)))
    values = counts / counts.sum() if normalize and counts.sum() > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, values, width=np.diff(edges), color=color, alpha=s.float('bar_alpha', 0.6),
           bottom=0, label=label)
    return values


def style_polar(ax, s):
    """下 = 0°、右 = 90° の向きにそろえる"""
    ax.set_theta_zero_location('S')
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(s.list('theta_labels', ['0°', '90°', '180°', '-90°']))
    ax.set_rlabel_position(s.float('rlabel_position', 135))


def set_common_rlim(axes, value_arrays):
    """複数の極座標ヒストグラムの r 軸を最大値 × 1.1 にそろえる"""
    max_val = max((v.max() for v in value_arrays if len(v)), default=0)
    r_max = max_val * 1.1 if max_val > 0 else 1
    for ax in axes:
        ax.set_rlim(0, r_max)


def annotated_grid(fig, spec, images, s):
    """アノテーション画像を spec（GridSpecの1区画）の中にタイル状に並べる"""
    n = len(images)
    if n == 0:
        fig.add_subplot(spec).axis('off')
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    inner = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=spec, wspace=0.1, hspace=0.1)
    for i, image in enumerate(images):
        ax = fig.add_subplot(inner[i])
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(s.str('annotated_title', 'Annotated {i}').format(i=i + 1),
                     fontsize=s.float('annotated_title_fontsize', 10))
        ax.axis('off')


def tile_figures(items, s, title_key, per_figure=10, cols=5):
    """個々の馬蹄形のデバッグ画像を per_figure 枚ずつの図にして返す"""
    figures = []
    for start in range(0, len(items), per_figure):
        chunk = items[start:start + per_figure]
        rows = math.ceil(len(chunk) / cols)
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        fig.suptitle(s.str(title_key).format(n=start // per_figure + 1), fontsize=14)
        for i, item in enumerate(chunk):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(item['image'])
            side = 'Ventral' if item['side'] == 'left' else 'Dorsal'
            ax.set_title(f"{side} | {item['angle']:.1f}°", fontsize=10)
            ax.axis('off')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figures.append(fig)
    return figures
