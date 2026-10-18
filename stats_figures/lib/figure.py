"""図の保存と「文字無し」版の出力（shared/figure.py に stats_figures の出力先と --dataset を付けたもの）"""
from shared.figure import NOTEXT_SUFFIX, finish, strip_text  # noqa: F401
from shared.figure import make_parser as _make_parser
from shared.figure import save_figure as _save_figure

from .paths import OUTPUT_DIR
from .settings import dataset_names


def make_parser(description, name, dataset_default=None):
    """description（スクリプトの docstring）をそのまま -h に表示する ArgumentParser を作る"""
    parser = _make_parser(description, name)
    if dataset_default is not None:
        parser.add_argument('--dataset', default=dataset_default, choices=dataset_names(),
                            help=f'config/datasets.txt のセクション名 (既定: {dataset_default})')
    return parser


def save_figure(fig, name, args, settings, subdir, text_free=True):
    """output/<subdir>/<name>.<format> に保存する（文字無し版は同じ範囲で切り出し）"""
    return _save_figure(fig, name, args, settings, OUTPUT_DIR / subdir, text_free=text_free)
