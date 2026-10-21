"""
論文提出用の図の見た目を統一するための共通スタイル・プロット関数。

背景:
    Detectron2/inference_ver2_heatmaps.py, heatmap/mosaic_horseshoe.py,
    Cellpose/inference_test.py, Cellpose/inference_nd2_zstack.py は
    それぞれ独立したパイプラインだが、最終的に人が見る図（V/Dの平均輝度
    ヒートマップ、検出結果のオーバーレイ）は同じ見た目・同じ約束事で
    統一したい。このモジュールはその共通部分（フォント、タイトル、
    カラーバー、パネルラベルなど）を1箇所にまとめる。

重要な設計判断（左右ヒートマップの共有カラースケール）:
    左右（V/D）のヒートマップをそれぞれ独立に0-255へmin-max正規化すると、
    片側が全体的に暗くても「そのヒートマップの中で一番明るいピクセル」が
    255になってしまい、V/D間の絶対的な明るさの違いが図から読み取れなく
    なる（比較のための図で、比較できない色になってしまう）。
    plot_vd_heatmap_figure() は、左右で同じ vmin/vmax（共有カラースケール）
    を使うことでこれを避ける。呼び出し側は正規化前の平均輝度をそのまま
    渡すこと。

使い方:
    from figure_style import plot_single_heatmap_figure, plot_vd_heatmap_figure

    # 単体のヒートマップ
    fig, ax = plot_single_heatmap_figure(heatmap, title="Ventral (V)", n=12)
    plt.show()

    # 左右をまとめた比較図（共有カラースケール）
    fig, axes = plot_vd_heatmap_figure(heatmap_v, heatmap_d, n_v=12, n_d=15)
    plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CMAP = "viridis"
DEFAULT_COLORBAR_LABEL = "Mean intensity (a.u.)"


def set_publication_style():
    """
    論文投稿を想定したmatplotlibのグローバルスタイルを設定する。
    サンセリフ体・白背景・太字タイトルなど、投稿先で問われやすい体裁を
    一括で揃える。呼び出しは冪等（何度呼んでも同じ設定になる）。
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 110,
    })


def shared_intensity_range(*heatmaps, percentile=(1, 99)):
    """
    複数のヒートマップ配列から、共有のvmin/vmaxを計算する。
    単純なmin/maxではなく1-99パーセンタイルを使うのは、外れ値の
    ピクセル1つで色レンジ全体が潰れる（コントラストが失われる）のを
    防ぐため。
    """
    values = np.concatenate([
        np.asarray(h, dtype=np.float64).ravel()
        for h in heatmaps if h is not None and np.asarray(h).size > 0
    ])
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, percentile)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def plot_heatmap_panel(ax, heatmap, *, vmin, vmax, cmap=DEFAULT_CMAP,
                        title=None, n=None, panel_label=None,
                        colorbar=True, colorbar_label=DEFAULT_COLORBAR_LABEL):
    """
    1枚のヒートマップを論文品質のスタイルで1つのAxesに描画する。
    単体出力・左右比較図の両方から共通で使われる部品。
    """
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    if title is not None:
        full_title = title if n is None else f"{title} (n = {n})"
        ax.set_title(full_title, pad=8)

    if panel_label is not None:
        ax.text(-0.05, 1.08, panel_label, transform=ax.transAxes,
                 fontsize=15, fontweight="bold", va="bottom", ha="right")

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)
        cbar.outline.set_visible(False)

    return im


def plot_single_heatmap_figure(heatmap, *, vmin=None, vmax=None, cmap=DEFAULT_CMAP,
                                title=None, n=None,
                                colorbar_label=DEFAULT_COLORBAR_LABEL,
                                figsize=(5, 4.2)):
    """
    ヒートマップ単体を論文品質で出力する（個別図として使う場合）。
    vmin/vmaxを省略すると、そのヒートマップ自身の1-99パーセンタイルで
    自動スケーリングする。
    """
    set_publication_style()
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = shared_intensity_range(heatmap)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_heatmap_panel(ax, heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
                        title=title, n=n, colorbar=True, colorbar_label=colorbar_label)
    return fig, ax


def plot_vd_heatmap_figure(heatmap_v, heatmap_d, *, n_v=None, n_d=None,
                            cmap=DEFAULT_CMAP, vmin=None, vmax=None,
                            colorbar_label=DEFAULT_COLORBAR_LABEL,
                            v_label="Ventral (V)", d_label="Dorsal (D)",
                            suptitle=None, figsize=(9, 4.2)):
    """
    V/D（左右）の平均輝度ヒートマップを、左右で共有するカラースケールで
    並べた比較図を作る。vmin/vmaxを省略すると、両方のヒートマップを
    合わせた1-99パーセンタイルから自動計算する。
    """
    set_publication_style()

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = shared_intensity_range(heatmap_v, heatmap_d)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im = plot_heatmap_panel(axes[0], heatmap_v, vmin=vmin, vmax=vmax, cmap=cmap,
                             title=v_label, n=n_v, panel_label="A", colorbar=False)
    plot_heatmap_panel(axes[1], heatmap_d, vmin=vmin, vmax=vmax, cmap=cmap,
                        title=d_label, n=n_d, panel_label="B", colorbar=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label(colorbar_label)
    cbar.outline.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    return fig, axes
