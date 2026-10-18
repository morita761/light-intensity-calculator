"""モザイク解析: 通常マスクとRFPマスクの平均GFP輝度ヒートマップを比較（2行2列）

用途: モザイク個体で、通常マスク（全馬蹄形）と RFP マスク（RFP陽性 = 遺伝子操作した細胞の
      馬蹄形）の GFP 輝度分布を比べる。明るさの scaling は通常マスクから計算し、
      両方とも通常マスクの最大値を 255 としてそろえる（同じスケールで比較できる）。
      旧 test_mosaic.py を整理したもの。
入力: config/image_sets.txt の4列（元画像 | 通常マスク | RFP画像 | RFPマスク）のセット
出力: output/mosaic_compare/<セット名>_overview.png  アノテーション(RFP画像上) + 4枚のヒートマップ
      output/mosaic_compare/<セット名>_grid.png      4枚のヒートマップ（2行2列、共通カラーバー）

Usage:
    python mosaic_compare.py --set mosaic_control
    python mosaic_compare.py --set mosaic_fz --both --no-show
"""
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from lib.core import (MOSAIC_SIDE_COLORS, SIDES, ExtractParams, average_heatmap, collect_image_set,
                      global_mean_intensity, to_uint8_by_max)
from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.plots import annotated_grid, heatmap_panel
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import check_files, get_image_set, imwrite
from shared.settings import load_figure_settings

NAME = 'mosaic_compare'
PANELS = (('normal', 'left'), ('normal', 'right'), ('rfp', 'left'), ('rfp', 'right'))


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    mosaic_sets = [n for n, st in sets.items() if st.pairs and all('rfp_mask' in p for p in st.pairs)]
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"4列そろったセット: {', '.join(mosaic_sets) or '（なし）'}"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--set', default=s.str('default_set') or None, choices=list(sets),
                        help='config/image_sets.txt のセット名（4列そろっているもの）')
    parser.add_argument('--center-margin', type=int, default=None,
                        help=f"中心線から何px以内の馬蹄形を除外するか（既定: {s.int('center_margin', 10)}）")
    args = parser.parse_args()
    if args.set is None:
        parser.error('--set を指定してください')

    image_set = get_image_set(sets, args.set)
    if not all('rfp' in p and 'rfp_mask' in p for p in image_set.pairs):
        sys.exit(f"エラー: [{args.set}] は 元画像 | マスク | RFP画像 | RFPマスク の4列で書いてください。")
    missing = check_files(image_set.pairs, ('original', 'mask', 'rfp', 'rfp_mask'))
    for path in missing:
        print(f"警告: 見つかりません（スキップ）: {path}")

    params = ExtractParams.from_settings(s, center_margin=args.center_margin)

    # Global Mean は通常マスク + RFPマスクの両方の画素から
    global_mean = global_mean_intensity(image_set.pairs, params.colors, mask_keys=('mask', 'rfp_mask'))
    print(f"全体の基準平均輝度 (Global Mean): {global_mean:.2f}")

    print("\n--- 通常マスク ---")
    rois_normal, debug_images = collect_image_set(
        image_set.pairs, global_mean, params, annotate_key='rfp',
        side_colors=MOSAIC_SIDE_COLORS, overlay_alpha=s.float('overlay_alpha', 0.4))
    print("\n--- RFPマスク ---")
    rois_rfp, _ = collect_image_set(image_set.pairs, global_mean, params, mask_key='rfp_mask', annotate=False)
    if not debug_images:
        sys.exit("エラー: 読み込めた画像がありません。config/image_sets.txt と pics/ を確認してください。")

    avg = {('normal', side): average_heatmap(rois_normal[side], params.target_size, label=f'Normal {side}')
           for side in SIDES}
    avg.update({('rfp', side): average_heatmap(rois_rfp[side], params.target_size, label=f'RFP {side}')
                for side in SIDES})

    # 通常マスクの左右の最大値で両方をそろえる
    normal_max = max(float(np.max(avg[('normal', side)])) for side in SIDES) or 1.0
    print(f"通常マスク ヒートマップの最大値: {normal_max:.2f}（RFPマスクの正規化にも使用）")
    heat = {key: to_uint8_by_max(value, normal_max) for key, value in avg.items()}

    out_dir = OUTPUT_DIR / NAME
    for i, image in enumerate(debug_images, start=1):
        imwrite(out_dir / image_set.name / f'annotated_{i:02d}.png', image)
    fmt = {'label': image_set.label, 'max': normal_max}

    # 1. アノテーション + 4枚
    fig = plt.figure(figsize=s.floats('overview_figsize', (16, 4)))
    outer = gridspec.GridSpec(1, 5, figure=fig)
    annotated_grid(fig, outer[0, 0], debug_images, s)
    for col, (mask_kind, side) in enumerate(PANELS, start=1):
        heatmap_panel(fig, fig.add_subplot(outer[0, col]), heat[(mask_kind, side)], s,
                      f'{mask_kind}_{side}_title', colorbar=False, title_fmt=fmt)
    fig.tight_layout()
    save_figure(fig, f'{image_set.name}_overview', args, s, out_dir)

    # 2. 2行2列 + 共通カラーバー
    fig, axes = plt.subplots(2, 2, figsize=s.floats('grid_figsize', (6, 4)), constrained_layout=True)
    im = None
    for ax, (mask_kind, side) in zip(axes.ravel(), PANELS):
        im = heatmap_panel(fig, ax, heat[(mask_kind, side)], s, f'{mask_kind}_{side}_title',
                           colorbar=False, title_fmt=fmt)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.75)
    cbar.set_label(s.str('colorbar_label').format(**fmt))
    save_figure(fig, f'{image_set.name}_grid', args, s, out_dir)

    finish(args)


if __name__ == '__main__':
    main()
