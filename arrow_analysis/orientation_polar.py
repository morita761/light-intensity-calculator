"""1つの画像セットの馬蹄形の向き（開口部の方向）を極座標ヒストグラムで表示

用途: 各馬蹄形の凹（開口部）が向いている方向を推定し、Ventral / Dorsal 別に
      角度の分布を極座標ヒストグラムにする（下 = 0°、右 = 90°、左 = -90°）。
      アノテーション画像には推定した向きを矢印で描く。
      旧 detect_arrow_improved.py / detect_arrow.py / orientation-polar-histogram.py を統合したもの。
        --method skeleton  骨格化して端点の中点から推定（既定、旧 detect_arrow_improved.py）
        --method moment    重心 → バウンディングボックス中心（旧 detect_arrow.py 等）
        --count            縦軸を割合でなく個数にする（旧 orientation-polar-histogram.py）
        --debug-rois       馬蹄形ごとの拡大画像 + 矢印も出力
        --debug-skeleton   骨格・端点（緑）・分岐点（赤）の画像も出力（skeleton のみ）
入力: config/image_sets.txt の [セット名] の元画像・マスク（pics/ 以下）
出力: output/orientation_polar/<セット名>_polar.png      アノテーション画像 + 左右の極座標ヒストグラム
      output/orientation_polar/<セット名>/annotated_*.png  矢印を描いた画像
      output/orientation_polar/<セット名>_angles.csv     全馬蹄形の角度

Usage:
    python orientation_polar.py --set R8control
    python orientation_polar.py --set R8ori --method moment --center-margin 50
    python orientation_polar.py --set R8ori --debug-rois --debug-skeleton --no-show
"""
import csv
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lib.orientation import METHODS, OrientationParams, collect_angles
from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.plots import annotated_grid, polar_hist, set_common_rlim, style_polar, tile_figures
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import get_image_set, imwrite
from shared.settings import load_figure_settings

NAME = 'orientation_polar'


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"画像セット: config/image_sets.txt（定義済み: {', '.join(sets)}）"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--set', default=s.str('default_set') or None, choices=list(sets),
                        help='config/image_sets.txt のセット名')
    parser.add_argument('--method', choices=METHODS, default=None,
                        help=f"向きの推定方法（既定: {s.str('method', 'skeleton')}）")
    parser.add_argument('--center-margin', type=int, default=None,
                        help=f"中心線から何px以内の馬蹄形を除外するか（既定: {s.int('center_margin', 80)}）")
    parser.add_argument('--count', action='store_true', help='縦軸を割合でなく個数にする')
    parser.add_argument('--debug-rois', action='store_true', help='馬蹄形ごとの拡大画像 + 矢印も出力')
    parser.add_argument('--debug-skeleton', action='store_true', help='骨格・端点・分岐点の画像も出力')
    args = parser.parse_args()
    if args.set is None:
        parser.error('--set を指定してください')

    image_set = get_image_set(sets, args.set)
    params = OrientationParams.from_settings(s, method=args.method, center_margin=args.center_margin)
    print(f"[{image_set.name}] {image_set.label}: {len(image_set.pairs)} 組, "
          f"方法 {params.method}, 中心線除外 {params.center_margin}px")

    result = collect_angles(image_set.pairs, params, debug_rois=args.debug_rois,
                            debug_skeleton=args.debug_skeleton)
    if not result['annotated']:
        sys.exit("エラー: 読み込めた画像がありません。config/image_sets.txt と pics/ を確認してください。")
    angles = result['angles']
    print(f"合計: Ventral {len(angles['left'])} 個, Dorsal {len(angles['right'])} 個")

    out_dir = OUTPUT_DIR / NAME
    tag = f'{image_set.name}_{params.method}'
    for i, image in enumerate(result['annotated'], start=1):
        imwrite(out_dir / image_set.name / f'annotated_{params.method}_{i:02d}.png', image)
    with open(out_dir / f'{tag}_angles.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['side', 'angle_deg'])
        for side, label in (('left', 'Ventral'), ('right', 'Dorsal')):
            writer.writerows([label, f'{a:.3f}'] for a in angles[side])

    # アノテーション画像 + 左右の極座標ヒストグラム
    fig = plt.figure(figsize=s.floats('figsize', (12, 4)))
    outer = gridspec.GridSpec(1, 3, figure=fig)
    annotated_grid(fig, outer[0, 0], result['annotated'], s)
    polar_axes, values = [], []
    for col, side, prefix, color in ((1, 'left', 'ventral_', s.str('ventral_color', 'magenta')),
                                     (2, 'right', 'dorsal_', s.str('dorsal_color', 'blue'))):
        ax = fig.add_subplot(outer[0, col], projection='polar')
        values.append(polar_hist(ax, angles[side], s, color, normalize=not args.count))
        style_polar(ax, s)
        if angles[side]:
            ax.set_title(s.str(prefix + 'title').format(label=image_set.label, n=len(angles[side])),
                         va='bottom', fontsize=s.float('title_fontsize'))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        polar_axes.append(ax)
    set_common_rlim(polar_axes, values)
    fig.tight_layout()
    save_figure(fig, f'{tag}_polar', args, s, out_dir)

    for key, items in (('rois', result['rois']), ('skeletons', result['skeletons'])):
        for n, fig in enumerate(tile_figures(items, s, f'{key}_title'), start=1):
            save_figure(fig, f'{tag}_{key}_{n:02d}', args, s, out_dir)

    finish(args)


if __name__ == '__main__':
    main()
