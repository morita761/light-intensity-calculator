"""ヒートマップ用の作図部品"""
import math

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def heatmap_panel(fig, ax, data, s, title_key, colorbar=True, title_fmt=None):
    """平均ヒートマップを1枚描く（cmap / vmin / vmax / colorbar_label は設定から）"""
    im = ax.imshow(data, cmap=s.str('cmap', 'viridis'), vmin=s.float('vmin', 0), vmax=s.float('vmax', 255))
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=s.float('colorbar_shrink', 1.0))
        cbar.set_label(s.str('colorbar_label'), fontsize=s.float('label_fontsize'))
    title = s.str(title_key)
    if title_fmt:
        title = title.format(**title_fmt)
    ax.set_title(title, fontsize=s.float('title_fontsize'))
    ax.axis('off')
    return im


def annotated_grid(fig, spec, images, s):
    """アノテーション画像を spec（GridSpecの1区画）の中にタイル状に並べる"""
    n = len(images)
    if n == 0:
        ax = fig.add_subplot(spec)
        ax.axis('off')
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


def overview_figure(debug_images, heat_left, heat_right, s, title_fmt=None):
    """[アノテーション画像 | Ventral ヒートマップ | Dorsal ヒートマップ] の1行3列"""
    fig = plt.figure(figsize=s.floats('overview_figsize', (12, 4)))
    outer = gridspec.GridSpec(1, 3, figure=fig)
    annotated_grid(fig, outer[0, 0], debug_images, s)
    heatmap_panel(fig, fig.add_subplot(outer[0, 1]), heat_left, s, 'ventral_title', title_fmt=title_fmt)
    heatmap_panel(fig, fig.add_subplot(outer[0, 2]), heat_right, s, 'dorsal_title', title_fmt=title_fmt)
    fig.tight_layout()
    return fig


def single_heatmap_figure(data, s, title_key, title_fmt=None):
    """ヒートマップ1枚 + カラーバー（論文用の個別図）"""
    fig, ax = plt.subplots(figsize=s.floats('single_figsize', (5, 4)))
    heatmap_panel(fig, ax, data, s, title_key, title_fmt=title_fmt)
    fig.tight_layout()
    return fig


def flatten(arrays):
    return np.concatenate([np.asarray(a).ravel() for a in arrays]) if len(arrays) else np.array([])
