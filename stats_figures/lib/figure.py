"""図の保存と「文字無し」版の出力

全ての作図スクリプトで共通のコマンドライン引数:
    --no-text  文字無し版（タイトル・軸ラベル・目盛りラベル・凡例・注釈を消したもの）だけ保存
    --both     通常版と文字無し版を両方保存
    --no-show  画面に表示せず保存だけする
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .paths import OUTPUT_DIR
from .settings import dataset_names

NOTEXT_SUFFIX = '_notext'


def make_parser(description, name, dataset_default=None):
    """description（スクリプトの docstring）をそのまま -h に表示する ArgumentParser を作る"""
    epilog = [f"軸・タイトル等の設定: config/figure_settings.txt の [{name}]",
              f"出力先: output/{name}/"]
    parser = argparse.ArgumentParser(description=description, epilog='\n'.join(epilog),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    if dataset_default is not None:
        datasets = dataset_names()
        parser.add_argument('--dataset', default=dataset_default, choices=datasets,
                            help=f'config/datasets.txt のセクション名 (既定: {dataset_default})')
    text = parser.add_mutually_exclusive_group()
    text.add_argument('--no-text', action='store_true',
                      help='文字無し版のみ保存（ファイル名末尾に _notext）')
    text.add_argument('--both', action='store_true',
                      help='通常版と文字無し版を両方保存')
    parser.add_argument('--no-show', action='store_true', help='画面表示をしない')
    return parser


def strip_text(fig):
    """図中の文字を全て消す（軸線・目盛り線・プロット・有意差の線は残す）"""
    for ax in fig.axes:
        for loc in ('center', 'left', 'right'):
            ax.set_title('', loc=loc)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for text in list(ax.texts):
            text.remove()
    for legend in list(fig.legends):
        legend.remove()
    for text in list(fig.texts):  # suptitle を含む
        text.remove()


def save_figure(fig, name, args, settings, subdir, text_free=True):
    """output/<subdir>/<name>.<format> に保存する

    文字無し版は通常版と同じ範囲(bbox)で切り出すので、重ね合わせて位置が一致する。
    text_free=False の図（表など）は --no-text 指定時も通常版を保存する。
    """
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.str('format', 'png')
    dpi = settings.int('dpi', 300)
    savefig_kwargs = dict(dpi=dpi, facecolor='white')

    want_text = not getattr(args, 'no_text', False) or not text_free
    want_notext = text_free and (getattr(args, 'no_text', False) or getattr(args, 'both', False))

    # 文字ありの状態で切り出し範囲を決めておく
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)

    saved = []
    if want_text:
        path = out_dir / f'{name}.{fmt}'
        fig.savefig(path, bbox_inches=bbox, **savefig_kwargs)
        saved.append(path)
    if want_notext:
        strip_text(fig)
        path = out_dir / f'{name}{NOTEXT_SUFFIX}.{fmt}'
        fig.savefig(path, bbox_inches=bbox, **savefig_kwargs)
        saved.append(path)

    for path in saved:
        print(f"保存: {Path(path).relative_to(OUTPUT_DIR.parent)}")
    return saved


def finish(args):
    """全ての図を保存した後に呼ぶ"""
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
