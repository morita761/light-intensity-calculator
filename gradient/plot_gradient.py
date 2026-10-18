"""マスク画像の色付き長方形内の緑色輝度グラディエント（左 → 右の輝度変化）をプロット

用途: 細胞などに沿って長方形（青・黄など）を描いたマスク画像から、元画像の
      緑チャネルを列ごとに平均した輝度プロファイルを取り出し、勾配を比べる。
        single : 1組ずつ、各長方形のプロファイル（各最大値で0〜1）と、その平均を出力
        multi  : 複数組の平均プロファイルを重ねる（全画像共通の最大値で0〜1）
      画像が1組だけのセットは自動で single になる。
入力: config/image_sets.txt の [セット名]（元画像 | マスク の2列、pics/ 以下）
出力: output/plot_gradient/<セット名>_<番号>_regions.png   各長方形のプロファイル（single）
      output/plot_gradient/<セット名>_<番号>_average.png   平均プロファイル（single）
      output/plot_gradient/<セット名>_multi.png            複数組の平均プロファイル（multi）
      output/plot_gradient/<セット名>/rects_*.png          検出した長方形を描いた画像

Usage:
    python make_dummy_gradient.py                      # サンプル画像を作る（初回のみ）
    python plot_gradient.py --set dummy
    python plot_gradient.py --set control --colors blue yellow
    python plot_gradient.py --set control --mode single --both --no-show
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.profiles import (COLOR_RANGES, average_profile, extract_raw_profiles, interpolate_profiles, normalize_by,
                          normalize_each)
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import get_image_set, imwrite
from shared.settings import load_figure_settings

NAME = 'plot_gradient'


def line_figure(profiles, labels, s, prefix, cmap_name):
    fig, ax = plt.subplots(figsize=s.floats(prefix + 'figsize', (12, 7)))
    if s.has(prefix + 'color'):
        colors = [s.str(prefix + 'color')] * len(profiles)
    else:
        colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, max(len(profiles), 1)))
    for profile, label, color in zip(profiles, labels, colors):
        ax.plot(profile, color=color, label=label)
    if s.bool('legend', True):
        ax.legend()
    ax.grid(True)
    s.apply(ax, prefix=prefix)
    fig.tight_layout()
    return fig


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"画像セット: config/image_sets.txt（定義済み: {', '.join(sets)}）"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--set', default=s.str('default_set') or None, choices=list(sets),
                        help='config/image_sets.txt のセット名')
    parser.add_argument('--mode', choices=['auto', 'single', 'multi'], default='auto',
                        help='auto: 1組なら single、複数組なら multi')
    parser.add_argument('--colors', nargs='+', choices=list(COLOR_RANGES), default=s.list('colors', ['blue']),
                        help='長方形の色（複数指定可）')
    args = parser.parse_args()
    if args.set is None:
        parser.error('--set を指定してください')

    image_set = get_image_set(sets, args.set)
    mode = args.mode if args.mode != 'auto' else ('single' if len(image_set.pairs) == 1 else 'multi')
    min_area = s.int('min_area', 100)
    print(f"[{image_set.name}] {len(image_set.pairs)} 組, モード {mode}, 検出色 {args.colors}")

    raw = []
    for pair in image_set.pairs:
        profiles, img = extract_raw_profiles(pair['mask'], pair['original'], args.colors, min_area)
        if profiles is not None:
            raw.append((pair['original'].name, profiles, img))
    if not raw:
        sys.exit("エラー: 有効なプロファイルが取得できませんでした。")

    out_dir = OUTPUT_DIR / NAME
    for i, (_, _, img) in enumerate(raw, start=1):
        imwrite(out_dir / image_set.name / f'rects_{i:02d}.png', img)

    if mode == 'single':
        for i, (file_name, profiles, _) in enumerate(raw, start=1):
            print(f"  {file_name}: {len(profiles)} 領域")
            normalized = normalize_each(profiles)
            fig = line_figure(normalized, [f'Region {k + 1}' for k in range(len(normalized))], s,
                              'regions_', s.str('regions_cmap', 'jet'))
            save_figure(fig, f'{image_set.name}_{i:02d}_regions', args, s, out_dir)
            fig = line_figure([average_profile(normalized)], [s.str('average_label', 'Average Gradient')], s,
                              'average_', 'viridis')
            save_figure(fig, f'{image_set.name}_{i:02d}_average', args, s, out_dir)
    else:
        global_max = max(float(p.max()) for _, profiles, _ in raw for p in profiles)
        print(f"グローバル最大値: {global_max}")
        averages = [average_profile(normalize_by(profiles, global_max)) for _, profiles, _ in raw]
        averages = interpolate_profiles(averages, min(len(a) for a in averages))
        fig = line_figure(averages, [name for name, _, _ in raw], s, 'multi_', s.str('multi_cmap', 'tab10'))
        save_figure(fig, f'{image_set.name}_multi', args, s, out_dir)

    finish(args)


if __name__ == '__main__':
    main()
