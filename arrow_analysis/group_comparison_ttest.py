"""2群間で馬蹄形の向き（角度）を比較（Welch の t検定 + 箱ひげ図 + 極座標ヒストグラム）

用途: Control 群と Test 群（例: vang Cas9）で、馬蹄形の開口部の向きが変わるかを
      Ventral / Dorsal 別に Welch の t検定で比べる（角度: 下 = 0°、右 = 90°、左 = -90°）。
      向きは skeleton 法で推定する（orientation_polar.py と同じ）。
入力: config/image_sets.txt の --control と --test のセット
出力: output/group_comparison_ttest/<control>_vs_<test>.png
        上段: Ventral / Dorsal の箱ひげ図（t検定のアスタリスク付き）
        下段: Ventral / Dorsal の極座標ヒストグラム（2群を重ねる）
      output/group_comparison_ttest/<control>_vs_<test>_ttest.csv  検定結果

Usage:
    python group_comparison_ttest.py                                  # figure_settings.txt の既定
    python group_comparison_ttest.py --control R8control --test R8ori
    python group_comparison_ttest.py --control 29d_R8control --test 29d_R8ori --both --no-show
"""
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lib.orientation import METHODS, OrientationParams, collect_angles
from lib.paths import FIGURE_SETTINGS_FILE, IMAGE_SETS_FILE, OUTPUT_DIR, ROOT
from lib.plots import polar_hist, set_common_rlim, style_polar
from shared.cli import add_image_sets_arg, preload_image_sets
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import get_image_set
from shared.settings import load_figure_settings

NAME = 'group_comparison_ttest'
SIDE_LABELS = {'left': 'Ventral', 'right': 'Dorsal'}


def p_to_asterisks(p):
    if p is None or np.isnan(p):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def welch_ttest(control, test):
    result = {'control_n': len(control), 'test_n': len(test),
              'control_mean': np.nan, 'control_std': np.nan, 'test_mean': np.nan, 'test_std': np.nan,
              't_statistic': np.nan, 'p_value': np.nan}
    if len(control) < 2 or len(test) < 2:
        print(f"  警告: t検定にはデータが足りません（Control n={len(control)}, Test n={len(test)}）")
        return result
    result.update(control_mean=np.mean(control), control_std=np.std(control, ddof=1),
                  test_mean=np.mean(test), test_std=np.std(test, ddof=1))
    t_stat, p_value = stats.ttest_ind(control, test, equal_var=False)
    result.update(t_statistic=t_stat, p_value=p_value)
    return result


def main():
    sets_file, sets = preload_image_sets(IMAGE_SETS_FILE, ROOT)
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)

    parser = make_parser(__doc__, NAME, [f"画像セット: config/image_sets.txt（定義済み: {', '.join(sets)}）"])
    add_image_sets_arg(parser, sets_file)
    parser.add_argument('--control', default=s.str('control_set') or None, choices=list(sets),
                        help='Control 群のセット名')
    parser.add_argument('--test', default=s.str('test_set') or None, choices=list(sets),
                        help='Test 群のセット名')
    parser.add_argument('--method', choices=METHODS, default=None,
                        help=f"向きの推定方法（既定: {s.str('method', 'skeleton')}）")
    parser.add_argument('--center-margin', type=int, default=None,
                        help=f"中心線から何px以内の馬蹄形を除外するか（既定: {s.int('center_margin', 80)}）")
    args = parser.parse_args()
    if not (args.control and args.test):
        parser.error('--control と --test を指定してください')

    control_set = get_image_set(sets, args.control)
    test_set = get_image_set(sets, args.test)
    params = OrientationParams.from_settings(s, method=args.method, center_margin=args.center_margin)

    data = {}
    for key, st in (('control', control_set), ('test', test_set)):
        print(f"\n{'=' * 60}\n{st.label} ({st.name}): {len(st.pairs)} 組\n{'=' * 60}")
        data[key] = collect_angles(st.pairs, params, annotate=False)['angles']
        print(f"  合計: Ventral {len(data[key]['left'])} 個, Dorsal {len(data[key]['right'])} 個")
        if not (data[key]['left'] or data[key]['right']):
            sys.exit(f"エラー: [{st.name}] から馬蹄形が1つも取れませんでした。config/image_sets.txt と pics/ を確認してください。")

    print(f"\n{'=' * 70}\nT-TEST RESULTS (Welch)\n{'=' * 70}")
    results = {}
    for side, label in SIDE_LABELS.items():
        r = welch_ttest(data['control'][side], data['test'][side])
        results[side] = r
        print(f"\n{label} Side:")
        print(f"  {control_set.label}: n={r['control_n']}, mean={r['control_mean']:.2f}°, std={r['control_std']:.2f}°")
        print(f"  {test_set.label}: n={r['test_n']}, mean={r['test_mean']:.2f}°, std={r['test_std']:.2f}°")
        print(f"  t = {r['t_statistic']:.4f}, p = {r['p_value']:.10f} {p_to_asterisks(r['p_value'])}")

    out_dir = OUTPUT_DIR / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{control_set.name}_vs_{test_set.name}'
    with open(out_dir / f'{tag}_ttest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = list(results['left'])
        writer.writerow(['side', *keys])
        for side, label in SIDE_LABELS.items():
            writer.writerow([label, *[results[side][k] for k in keys]])

    labels = [control_set.label, test_set.label]
    colors = s.list('group_colors', ['blue', 'red'])
    box_colors = s.list('box_colors', ['lightblue', 'lightcoral'])
    fig = plt.figure(figsize=s.floats('figsize', (14, 10)))

    # 上段: 箱ひげ図
    for col, (side, label) in enumerate(SIDE_LABELS.items(), start=1):
        ax = fig.add_subplot(2, 2, col)
        r = results[side]
        bp = ax.boxplot([data['control'][side], data['test'][side]], tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        s.apply(ax, prefix='box_')
        ax.set_title(
            f"{label} Side\n"
            f"{labels[0]}: n={r['control_n']}, mean={r['control_mean']:.1f}° ± {r['control_std']:.1f}°\n"
            f"{labels[1]}: n={r['test_n']}, mean={r['test_mean']:.1f}° ± {r['test_std']:.1f}°",
            fontsize=s.float('title_fontsize'))
        line_y = s.float('bracket_y', 120)
        height = s.float('bracket_height', 5)
        ax.plot([1, 1, 2, 2], [line_y, line_y + height, line_y + height, line_y], lw=1.5, c='black')
        ax.text(1.5, line_y + 3, p_to_asterisks(r['p_value']), ha='center', va='bottom',
                fontsize=s.float('star_fontsize', 14))

    # 下段: 極座標ヒストグラム（2群を重ねる）
    polar_axes, values = [], []
    for col, (side, label) in enumerate(SIDE_LABELS.items(), start=3):
        ax = fig.add_subplot(2, 2, col, projection='polar')
        for key, group_label, color in (('control', labels[0], colors[0]), ('test', labels[1], colors[1])):
            values.append(polar_hist(ax, data[key][side], s, color, label=group_label))
        style_polar(ax, s)
        ax.set_title(s.str('polar_title', '{side} - Polar Histogram').format(side=label), va='bottom', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        polar_axes.append(ax)
    set_common_rlim(polar_axes, values)

    fig.tight_layout()
    save_figure(fig, tag, args, s, out_dir)
    finish(args)


if __name__ == '__main__':
    main()
