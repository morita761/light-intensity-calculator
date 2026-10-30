"""1つの画像セットの馬蹄形を重ね合わせた平均GFP輝度ヒートマップ（Ventral / Dorsal）

用途: 画像セット内の全馬蹄形を同じ大きさ（100x100）にそろえて画素ごとに平均し、
      馬蹄形のどこが明るいか（輝度の偏り）を Ventral / Dorsal 別に可視化する。
      画像ごとの明るさの違いは Global Mean / 画像平均 で補正する。
      旧 horseshoe-heatmaps.py / trim_test.py / mosaic_horseshoe.py を統合したもの。
        --trim N       トリム平均（旧 trim_test.py）
        --overlay-rfp  RFP画像の上にアノテーションを半透明で描く（旧 mosaic_horseshoe.py）
        --histogram    平均ヒートマップの輝度分布も出力
入力: config/image_sets.txt の [セット名] の元画像・マスク（pics/ 以下）
出力: output/horseshoe_heatmap/<セット名>_overview.png   アノテーション画像 + 左右ヒートマップ
      output/horseshoe_heatmap/<セット名>_ventral.png    Ventral ヒートマップ単体
      output/horseshoe_heatmap/<セット名>_dorsal.png     Dorsal ヒートマップ単体
      output/horseshoe_heatmap/<セット名>/annotated_*.png 検出結果を描いた画像

Usage:
    python horseshoe_heatmap.py --set vang_cas9
    python horseshoe_heatmap.py --set vang_cas9 --trim 1 --center-margin 50
    python horseshoe_heatmap.py --set mosaic_fz --overlay-rfp --histogram
    python horseshoe_heatmap.py --set control --both --no-show
"""
import sys

import matplotlib.pyplot as plt

from lib.core import (MOSAIC_SIDE_COLORS, SIDE_COLORS, ExtractParams, average_heatmap, collect_image_set,
                      global_mean_intensity, to_uint8_minmax)
from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.plots import flatten, overview_figure, single_heatmap_figure
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import check_files, get_image_set, imwrite
from shared.settings import load_figure_settings

NAME = 'horseshoe_heatmap'


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"画像セット: config/image_sets.txt（定義済み: {', '.join(sets)}）"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--set', default=s.str('default_set') or None, choices=list(sets),
                        help='config/image_sets.txt のセット名')
    parser.add_argument('--trim', type=int, default=s.int('trim', 0),
                        help='トリム平均で上下それぞれ除外する個数（0 = 通常の平均）')
    parser.add_argument('--center-margin', type=int, default=None,
                        help=f"中心線から何px以内の馬蹄形を除外するか（既定: {s.int('center_margin', 10)}）")
    parser.add_argument('--overlay-rfp', action='store_true',
                        help='RFP画像（image_sets.txt の3列目）の上にアノテーションを半透明で描く')
    parser.add_argument('--histogram', action='store_true', help='平均ヒートマップの輝度分布も出力')
    args = parser.parse_args()
    if args.set is None:
        parser.error('--set を指定してください')

    image_set = get_image_set(sets, args.set)
    keys = ('original', 'mask', 'rfp') if args.overlay_rfp else ('original', 'mask')
    missing = check_files(image_set.pairs, keys)
    if missing:
        print("警告: 見つからないファイルがあります（スキップします）:")
        for path in missing:
            print(f"  {path}")

    params = ExtractParams.from_settings(s, center_margin=args.center_margin)
    print(f"[{image_set.name}] {image_set.label}: {len(image_set.pairs)} 組, "
          f"中心線除外 {params.center_margin}px, trim={args.trim}")

    global_mean = global_mean_intensity(image_set.pairs, params.colors)
    print(f"全体の基準平均輝度 (Global Mean): {global_mean:.2f}")

    rois, debug_images = collect_image_set(
        image_set.pairs, global_mean, params,
        annotate_key='rfp' if args.overlay_rfp else 'original',
        side_colors=MOSAIC_SIDE_COLORS if args.overlay_rfp else SIDE_COLORS,
        overlay_alpha=s.float('overlay_alpha', 0.4) if args.overlay_rfp else None)
    if not debug_images:
        sys.exit("エラー: 読み込めた画像がありません。config/image_sets.txt と pics/ を確認してください。")

    heat = {side: to_uint8_minmax(average_heatmap(rois[side], params.target_size, args.trim, label))
            for side, label in (('left', 'Ventral'), ('right', 'Dorsal'))}

    out_dir = OUTPUT_DIR / NAME
    for i, image in enumerate(debug_images, start=1):
        imwrite(out_dir / image_set.name / f'annotated_{i:02d}.png', image)

    fmt = {'label': image_set.label}
    tag = image_set.name + (f'_trim{args.trim}' if args.trim else '')
    save_figure(overview_figure(debug_images, heat['left'], heat['right'], s, fmt), f'{tag}_overview', args, s, out_dir)
    save_figure(single_heatmap_figure(heat['left'], s, 'ventral_title', fmt), f'{tag}_ventral', args, s, out_dir)
    save_figure(single_heatmap_figure(heat['right'], s, 'dorsal_title', fmt), f'{tag}_dorsal', args, s, out_dir)

    if args.histogram:
        fig, axes = plt.subplots(1, 2, figsize=s.floats('histogram_figsize', (10, 5)))
        for ax, side, prefix, color in ((axes[0], 'left', 'histogram_ventral_', 'red'),
                                        (axes[1], 'right', 'histogram_dorsal_', 'blue')):
            ax.hist(flatten([heat[side]]), bins=s.int('histogram_bins', 50), range=(0, 255), color=color, alpha=0.6)
            ax.grid(axis='y', alpha=0.5)
            s.apply(ax, prefix=(prefix, 'histogram_'))
        fig.tight_layout()
        save_figure(fig, f'{tag}_histogram', args, s, out_dir)

    finish(args)


if __name__ == '__main__':
    main()
