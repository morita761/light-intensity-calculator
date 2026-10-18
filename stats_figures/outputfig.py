"""系統×V/Dごとの箱ひげ図を1枚ずつ出力（Ventral branch vs Dorsal branch）

用途: 各系統・各R8（Ventral/Dorsal）で、Ventral branch と Dorsal branch の
      輝度に差があるかを t検定付きの白黒箱ひげ図で示す（論文用の個別図）。
      例: two_groups なら Control_V, Control_D, Cas9_V, Cas9_D の4枚。
入力: data/processed/<dataset>/*_v.csv, *_d.csv
出力: output/outputfig/outputfig_<dataset>_<系統>_<V|D>.png

Usage:
    python outputfig.py
    python outputfig.py --dataset three_groups --both --no-show
"""
import matplotlib.pyplot as plt

from lib.data_io import load_vd
from lib.figure import finish, make_parser, save_figure
from lib.plots import vd_boxplot
from lib.settings import load_figure_settings

NAME = 'outputfig'


def main():
    args = make_parser(__doc__, NAME, dataset_default='two_groups').parse_args()
    s = load_figure_settings(NAME)

    for genotype, by_side in load_vd(args.dataset).items():
        for sec, prefix in (('V', 'ventral_'), ('D', 'dorsal_')):
            fig, ax = plt.subplots(figsize=s.floats('figsize', (6, 5)))
            t_stat, p_val = vd_boxplot(ax, by_side[sec], s)
            s.apply(ax, prefix=prefix)

            # タイトルは図の下中央に表示
            caption = s.str(prefix + 'caption')
            if caption:
                fig.text(0.5, 0.02, caption, ha='center', va='center',
                         fontsize=s.float('caption_fontsize', 17))
            fig.tight_layout(rect=[0, 0.05, 1, 1])

            print(f"[{genotype} {sec}] t = {t_stat:.4f}, p = {p_val:.5f}")
            safe = genotype.replace(' ', '_')
            save_figure(fig, f'{NAME}_{args.dataset}_{safe}_{sec}', args, s, NAME)
    finish(args)


if __name__ == '__main__':
    main()
