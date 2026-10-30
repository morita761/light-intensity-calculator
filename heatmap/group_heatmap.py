"""複数系統の平均GFP輝度ヒートマップを同じスケールで比較 + 輝度分布ヒストグラム

用途: Control / vang RNAi / vang Cas9 などの系統ごとに平均ヒートマップを作り、
      全系統・左右で共通の最大値を 255 としてそろえて比較する（系統間で明るさを比べられる）。
      Global Mean は全系統の画像をまとめて計算する。
      輝度分布は系統を重ねたヒストグラムで Ventral / Dorsal 別に比較する。
      旧 show-intensity-histgram.py / show-intensity-histgram-three-groups.py を統合したもの。
入力: config/image_sets.txt の各セット（--groups で指定、先頭から順に並ぶ）
出力: output/group_heatmap/<グループ名>_<セット名>.png   系統ごとの [アノテーション | Ventral | Dorsal]
      output/group_heatmap/<グループ名>_histogram.png    輝度分布の比較

Usage:
    python group_heatmap.py                                      # figure_settings.txt の groups
    python group_heatmap.py --groups control vang_rnai vang_cas9
    python group_heatmap.py --groups control vang_cas9 --hist-source average
    python group_heatmap.py --both --no-show
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from lib.core import SIDES, ExtractParams, average_heatmap, collect_image_set, global_mean_intensity, to_uint8_by_max
from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.plots import flatten, overview_figure
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import get_image_set
from shared.settings import load_figure_settings

NAME = 'group_heatmap'


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"画像セット: config/image_sets.txt（定義済み: {', '.join(sets)}）"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--groups', nargs='+', default=s.list('groups'), choices=list(sets),
                        help='比較する画像セット名（並べる順）')
    parser.add_argument('--center-margin', type=int, default=None,
                        help=f"中心線から何px以内の馬蹄形を除外するか（既定: {s.int('center_margin', 50)}）")
    parser.add_argument('--hist-source', choices=['raw', 'average'], default=s.str('hist_source', 'raw'),
                        help='ヒストグラムに使う値: raw = 全馬蹄形の全画素, average = 平均ヒートマップの画素')
    parser.add_argument('--name', default=None, help='出力ファイル名の頭（既定: セット名をつなげたもの）')
    args = parser.parse_args()
    if not args.groups:
        parser.error('--groups を指定してください')

    groups = [get_image_set(sets, g) for g in args.groups]
    params = ExtractParams.from_settings(s, center_margin=args.center_margin)

    # 1. 全系統まとめた Global Mean
    all_pairs = [pair for g in groups for pair in g.pairs]
    global_mean = global_mean_intensity(all_pairs, params.colors)
    print(f"全体の基準平均輝度 (Global Mean): {global_mean:.2f}")

    # 2. 系統ごとのROI収集と平均
    results = {}
    for g in groups:
        print(f"\n--- {g.label} ({g.name}) ---")
        rois, debug_images = collect_image_set(g.pairs, global_mean, params)
        averages = {side: average_heatmap(rois[side], params.target_size, label=f'{g.label} {side}')
                    for side in SIDES}
        results[g.name] = {'group': g, 'rois': rois, 'debug': debug_images, 'avg': averages}
    if not any(r['debug'] for r in results.values()):
        sys.exit("エラー: 読み込めた画像がありません。config/image_sets.txt と pics/ を確認してください。")

    # 3. 全系統・左右共通の最大値でそろえる
    global_max = max(float(np.max(r['avg'][side])) for r in results.values() for side in SIDES)
    print(f"\n全体の最大輝度値: {global_max:.2f}")

    tag = args.name or '_'.join(args.groups)
    out_dir = OUTPUT_DIR / NAME
    for r in results.values():
        g = r['group']
        heat = {side: to_uint8_by_max(r['avg'][side], global_max) for side in SIDES}
        r['heat'] = heat
        fig = overview_figure(r['debug'], heat['left'], heat['right'], s, {'label': g.label})
        save_figure(fig, f'{tag}_{g.name}', args, s, out_dir)

    # 4. 輝度分布ヒストグラム（系統を重ねる）
    colors = s.list('group_colors', ['blue', 'green', 'red', 'purple', 'orange'])
    fig, axes = plt.subplots(1, 2, figsize=s.floats('histogram_figsize', (14, 5)))
    for ax, side, prefix in ((axes[0], 'left', 'histogram_ventral_'), (axes[1], 'right', 'histogram_dorsal_')):
        values = {name: (flatten(r['rois'][side]) if args.hist_source == 'raw' else flatten([r['heat'][side]]))
                  for name, r in results.items()}
        max_val = max((v.max() for v in values.values() if v.size), default=1)
        for (name, v), color in zip(values.items(), colors * len(values)):
            if v.size:
                ax.hist(v, bins=s.int('histogram_bins', 50), range=(0, max_val), color=color, alpha=0.5,
                        label=results[name]['group'].label)
        ax.legend()
        ax.grid(axis='y', alpha=0.5)
        s.apply(ax, prefix=(prefix, 'histogram_'))
    fig.tight_layout()
    save_figure(fig, f'{tag}_histogram', args, s, out_dir)

    finish(args)


if __name__ == '__main__':
    main()
