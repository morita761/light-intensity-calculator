"""マスク画像の色付き長方形から、元画像の緑色輝度プロファイル（横方向）を取り出す

プロファイル = 長方形内の緑チャネルを列ごとに平均したもの（左 → 右の輝度変化）
"""
import cv2
import numpy as np

from shared.horseshoe import imread

# 検出可能な色（HSV範囲）。馬蹄形用より彩度・明度の下限が低い
COLOR_RANGES = {
    'blue': {'lower': np.array([90, 50, 50]), 'upper': np.array([130, 255, 255])},
    'yellow': {'lower': np.array([20, 50, 50]), 'upper': np.array([35, 255, 255])},
    'red': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255]),
            'lower2': np.array([160, 50, 50]), 'upper2': np.array([180, 255, 255])},
    'green': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
    'cyan': {'lower': np.array([80, 50, 50]), 'upper': np.array([100, 255, 255])},
    'magenta': {'lower': np.array([140, 50, 50]), 'upper': np.array([160, 255, 255])},
}


def combined_color_mask(hsv, color_names):
    mask = None
    for name in color_names:
        if name not in COLOR_RANGES:
            raise ValueError(f"未対応の色: {name}（対応色: {', '.join(COLOR_RANGES)}）")
        r = COLOR_RANGES[name]
        m = cv2.inRange(hsv, r['lower'], r['upper'])
        if 'lower2' in r:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, r['lower2'], r['upper2']))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def interpolate_profiles(profiles, target_length):
    """線形補間で全プロファイルを target_length の長さにそろえる → (N, target_length)"""
    x_target = np.linspace(0, 1, target_length)
    return np.array([np.interp(x_target, np.linspace(0, 1, len(p)), p) for p in profiles])


def average_profile(profiles):
    """最も短いプロファイルの長さにそろえて平均"""
    return interpolate_profiles(profiles, min(len(p) for p in profiles)).mean(axis=0)


def extract_raw_profiles(mask_path, original_path, colors, min_area=100):
    """返り値: (生プロファイルのリスト, 長方形を描いた元画像) または (None, None)"""
    mask_img = imread(mask_path)
    original = imread(original_path)
    if mask_img is None or original is None:
        print(f"エラー: 画像を読み込めませんでした: {mask_path if mask_img is None else original_path}")
        return None, None

    mask = combined_color_mask(cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV), colors)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    profiles = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        green = original[y:y + h, x:x + w, 1]
        profiles.append(np.mean(green, axis=0))
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if not profiles:
        print(f"エラー: {mask_path.name} に指定色({', '.join(colors)})の長方形が見つかりませんでした。"
              f"HSV範囲か面積の閾値を調整してください。")
        return None, None
    return profiles, original


def normalize_each(profiles):
    """各プロファイルをそれぞれの最大値で 0〜1 にする"""
    return [p / p.max() if p.max() > 0 else p for p in profiles]


def normalize_by(profiles, max_value):
    return [p / max_value if max_value > 0 else p for p in profiles]
