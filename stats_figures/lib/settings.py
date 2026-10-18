"""config/*.txt の読み込み

figure_settings.txt: 図ごとのタイトル・軸ラベル・軸範囲・サイズ等
datasets.txt       : 比較する系統（表示名 → 生データファイル）
"""
from collections import OrderedDict

from shared.settings import FigureSettings, read_config  # noqa: F401
from shared.settings import load_figure_settings as _load_figure_settings

from .paths import DATASETS_FILE, FIGURE_SETTINGS_FILE


def load_figure_settings(name):
    return _load_figure_settings(FIGURE_SETTINGS_FILE, name)


def load_dataset(name):
    """datasets.txt の [name] を {表示名: 生データファイル名} の順序付き辞書で返す"""
    parser = read_config(DATASETS_FILE)
    if not parser.has_section(name):
        available = ', '.join(parser.sections())
        raise KeyError(f"datasets.txt に [{name}] がありません（定義済み: {available}）")
    return OrderedDict((k, v.strip()) for k, v in parser[name].items())


def dataset_names():
    return read_config(DATASETS_FILE).sections()
