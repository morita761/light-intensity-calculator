"""設定ファイル（config/*.txt, INI形式）の読み込み

各フォルダの config/figure_settings.txt を [common] + [図の名前] でマージして使う。
"""
import configparser

from pathlib import Path


def read_config(path):
    """INI形式の txt を読む（キーの大文字・小文字を保持、% はそのまま使える）"""
    path = Path(path)
    # interpolation=None: タイトルやファイル名中の % をそのまま使えるように
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=None)
    parser.optionxform = str
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    parser.read(path, encoding='utf-8')
    return parser


class FigureSettings:
    """[common] と [図の名前] をマージした設定値

    値の書式:
        文字列 : そのまま（\\n で改行）
        数値   : 14
        組     : 10, 7
        ylim   : 0, 100 または auto
    """

    def __init__(self, values, name):
        self._values = values
        self.name = name

    def has(self, key):
        return key in self._values

    def str(self, key, default=''):
        if key not in self._values:
            return default
        return self._values[key].replace('\\n', '\n')

    def float(self, key, default=None):
        value = self._values.get(key, '').strip()
        return float(value) if value else default

    def int(self, key, default=None):
        value = self._values.get(key, '').strip()
        return int(value) if value else default

    def bool(self, key, default=False):
        value = self._values.get(key, '').strip().lower()
        if not value:
            return default
        return value in ('1', 'true', 'yes', 'on')

    def list(self, key, default=None):
        value = self._values.get(key, '').strip()
        if not value:
            return default
        return [v.strip() for v in value.split(',')]

    def floats(self, key, default=None):
        values = self.list(key)
        return tuple(float(v) for v in values) if values else default

    def ints(self, key, default=None):
        values = self.list(key)
        return tuple(int(v) for v in values) if values else default

    def ylim(self, key='ylim'):
        """'auto' や未設定なら None を返す"""
        value = self._values.get(key, 'auto').strip().lower()
        if value in ('', 'auto', 'none'):
            return None
        return self.floats(key)

    def apply(self, ax, prefix=''):
        """タイトル・軸ラベル・軸範囲・フォントサイズを ax に適用する

        prefix を指定すると '<prefix>title' のようなキーを優先して使う
        （1枚の図に複数パネルがある場合用）。タプルで渡すと前から順に探し、
        どれにもなければ接頭辞なしのキーを使う。
        """
        prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        candidates = [p for p in prefixes if p] + ['']

        def find(key):
            for p in candidates:
                if self.has(p + key):
                    return p + key
            return None

        def get(key):
            found = find(key)
            return self.str(found) if found else None

        title = get('title')
        if title is not None:
            ax.set_title(title, fontsize=self.float('title_fontsize'))
        xlabel = get('xlabel')
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.float('label_fontsize'))
        ylabel = get('ylabel')
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.float('label_fontsize'))

        ylim_key = find('ylim')
        ylim = self.ylim(ylim_key) if ylim_key else None
        if ylim is not None:
            ax.set_ylim(*ylim)

        tick_fontsize = self.float('tick_fontsize')
        if tick_fontsize:
            ax.tick_params(axis='both', labelsize=tick_fontsize)


def load_figure_settings(path, name):
    """path の [common] と [name] をマージした FigureSettings を返す"""
    parser = read_config(path)
    values = dict(parser['common']) if parser.has_section('common') else {}
    if parser.has_section(name):
        values.update(parser[name])
    else:
        print(f"注意: {Path(path).name} に [{name}] セクションがないため [common] のみ使用します。")
    return FigureSettings(values, name)
