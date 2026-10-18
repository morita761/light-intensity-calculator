"""マスク画像からの馬蹄形抽出（heatmap/, arrow_analysis/ 共通）

マスク画像: 元画像の上に馬蹄形を色（blue / cyan / magenta）で塗ったもの
元画像    : 輝度測定用（緑チャネル = GFP）

画像セットは各フォルダの config/image_sets.txt に書く（書式は load_image_sets を参照）。
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .settings import read_config

# アノテーションに使用する色のHSV範囲
ANNOTATION_COLORS_HSV = {
    'red': {
        'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255]),
        'lower2': np.array([170, 100, 100]), 'upper2': np.array([180, 255, 255]),
    },
    'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
    'cyan': {'lower': np.array([85, 100, 100]), 'upper': np.array([100, 255, 255])},
    'magenta': {'lower': np.array([140, 100, 100]), 'upper': np.array([170, 255, 255])},
}
DEFAULT_COLORS = ['blue', 'cyan', 'magenta']

PAIR_KEYS = ('original', 'mask', 'rfp', 'rfp_mask')


def imread(path):
    """cv2.imread と同じ（BGR）。日本語を含むパスでも読める。読めなければ None"""
    path = Path(path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, image):
    """cv2.imwrite と同じ。日本語を含むパスでも書ける"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or '.png', image)
    if ok:
        buf.tofile(str(path))
    return ok


def color_mask(mask_image_bgr, color_name):
    """指定色のHSV範囲に入る画素を 255 にした2値マスク"""
    hsv = cv2.cvtColor(mask_image_bgr, cv2.COLOR_BGR2HSV)
    ranges = ANNOTATION_COLORS_HSV[color_name]
    mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
    if 'lower2' in ranges:  # 赤のようにHueが2つの範囲にまたがる場合
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, ranges['lower2'], ranges['upper2']))
    return mask


def combined_color_mask(mask_image_bgr, colors=DEFAULT_COLORS):
    combined = np.zeros(mask_image_bgr.shape[:2], dtype=np.uint8)
    for color in colors:
        combined = cv2.bitwise_or(combined, color_mask(mask_image_bgr, color))
    return combined


def iter_horseshoe_contours(mask_image_bgr, colors=DEFAULT_COLORS, min_area=20):
    """色ごとに外側輪郭を検出し、面積 min_area 以上のものを順に返す"""
    for color in colors:
        contours, _ = cv2.findContours(color_mask(mask_image_bgr, color),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                yield contour


def image_center_x(image):
    """左右分割線 = 画像全体の中心X座標"""
    return image.shape[1] // 2


def classify_side(contour, center_x, margin):
    """輪郭を左右に分類する

    - バウンディングボックスが中心線をまたぐ → None（除外）
    - 中心X が 中心線 - margin より左 → 'left'（Ventral）
    - 中心X が 中心線 + margin より右 → 'right'（Dorsal）
    - それ以外（中心線に近すぎる） → None
    """
    x, _, w, _ = cv2.boundingRect(contour)
    if x < center_x < x + w:
        return None
    contour_center_x = x + w // 2
    if contour_center_x + margin < center_x:
        return 'left'
    if contour_center_x - margin > center_x:
        return 'right'
    return None


def filled_local_mask(contour):
    """輪郭のバウンディングボックス内を塗りつぶしたマスクと (x, y, w, h)"""
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local, [contour - np.array([x, y])], -1, 255, cv2.FILLED)
    return local, (x, y, w, h)


def dilate_to_occupancy(local_mask, occupancy, max_iterations=4):
    """バウンディングボックス内の占有率が occupancy を超えるまで 3x3 で膨張する

    返り値: (膨張後のマスク, 膨張回数)
    """
    kernel = np.ones((3, 3), np.uint8)
    total = local_mask.size
    iterations = 0
    mask = local_mask.copy()
    while True:
        rate = cv2.countNonZero(mask) / total if total > 0 else 0
        if rate > occupancy or iterations >= max_iterations:
            break
        mask = cv2.dilate(mask, kernel, iterations=1)
        iterations += 1
    return mask, iterations


def mean_in_mask(channel, mask):
    """mask 内で 0 より大きい画素の平均（なければ None）"""
    values = cv2.bitwise_and(channel, channel, mask=mask)
    values = values[values > 0]
    return float(np.mean(values)) if values.size else None


# ---------------------------------------------------------------------------
# 画像セット（config/image_sets.txt）
# ---------------------------------------------------------------------------

@dataclass
class ImageSet:
    name: str
    label: str
    pairs: list = field(default_factory=list)  # [{'original': Path, 'mask': Path, ...}]


def load_image_sets(path, base_dir):
    """image_sets.txt を読む

    書式:
        [common]
        image_root  = pics          # base_dir からの相対パス（画像の置き場所）
        mask_suffix = _mask.tif     # マスク名の省略時に使う接尾辞

        [セット名]
        label = Control             # グラフに表示する名前（省略時はセット名）
        dir   = control             # image_root からの相対フォルダ
        images =
            元画像.png                              ← マスク = 元画像名(拡張子なし) + mask_suffix
            元画像.png | マスク.tif
            元画像.png | マスク.tif | RFP画像.png | RFPマスク.tif   ← モザイク解析用

    返り値: {セット名: ImageSet}（記載順）
    """
    parser = read_config(path)
    common = parser['common'] if parser.has_section('common') else {}
    root = Path(base_dir) / common.get('image_root', 'pics')
    default_suffix = common.get('mask_suffix', '_mask.tif')

    sets = OrderedDict()
    for name in parser.sections():
        if name == 'common':
            continue
        section = parser[name]
        folder = root / section.get('dir', '')
        suffix = section.get('mask_suffix', default_suffix)
        pairs = []
        for line in section.get('images', '').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = [c.strip() for c in line.split('|')]
            if len(cols) == 1:
                cols.append(Path(cols[0]).stem + suffix)
            pairs.append({key: folder / value for key, value in zip(PAIR_KEYS, cols) if value})
        sets[name] = ImageSet(name=name, label=section.get('label', name), pairs=pairs)
    return sets


def get_image_set(sets, name, path_label='config/image_sets.txt'):
    if name not in sets:
        raise SystemExit(f"エラー: {path_label} に [{name}] がありません（定義済み: {', '.join(sets)}）")
    return sets[name]


def check_files(pairs, keys=('original', 'mask')):
    """存在しないファイルを列挙して返す"""
    missing = []
    for pair in pairs:
        for key in keys:
            if key in pair and not pair[key].exists():
                missing.append(pair[key])
    return missing
