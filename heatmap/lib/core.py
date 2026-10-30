"""馬蹄形ごとのGFP輝度ROIの抽出・輝度正規化・平均化

処理の流れ（全スクリプト共通）:
  1. 全画像のマスク内GFP輝度の平均 = Global Mean を求める
  2. 画像ごとに scaling = Global Mean / その画像のマスク内平均 で明るさをそろえる
  3. マスクの各馬蹄形を左右（Ventral/Dorsal）に分け、バウンディングボックスの
     GFP輝度（馬蹄形の外は0）を target_size にリサイズして集める
  4. 左右それぞれ平均（またはトリム平均）して平均ヒートマップにする
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from shared.horseshoe import (DEFAULT_COLORS, classify_side, combined_color_mask, dilate_to_occupancy,
                              filled_local_mask, image_center_x, imread, iter_horseshoe_contours,
                              mean_in_mask)

SIDES = ('left', 'right')
# アノテーション画像の塗り色 (BGR)
SIDE_COLORS = {'left': (0, 0, 255), 'right': (255, 0, 0)}         # 赤 / 青
MOSAIC_SIDE_COLORS = {'left': (255, 0, 255), 'right': (255, 0, 0)}  # マゼンタ / 青


@dataclass
class ExtractParams:
    """ROI抽出のパラメータ（figure_settings.txt から読む）"""
    center_margin: int = 10           # 中心線から何px以内の馬蹄形を除外するか
    dilate_occupancy: float = 0.6     # この占有率を超えるまで膨張（0 なら膨張しない）
    max_dilate: int = 2               # 膨張がこの回数を超えた馬蹄形は除外
    clip_uint8: bool = True           # 正規化後に 0-255 でクリップ（False なら float のまま）
    target_size: tuple = (100, 100)
    min_area: int = 20
    colors: list = field(default_factory=lambda: list(DEFAULT_COLORS))

    @classmethod
    def from_settings(cls, s, **overrides):
        params = cls(
            center_margin=s.int('center_margin', cls.center_margin),
            dilate_occupancy=s.float('dilate_occupancy', cls.dilate_occupancy),
            max_dilate=s.int('max_dilate', cls.max_dilate),
            clip_uint8=s.bool('clip_uint8', cls.clip_uint8),
            target_size=s.ints('target_size', cls.target_size),
            min_area=s.int('min_area', cls.min_area),
            colors=s.list('annotation_colors', list(DEFAULT_COLORS)),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(params, key, value)
        return params


def global_mean_intensity(pairs, colors, mask_keys=('mask',)):
    """全画像のマスク内（0より大きい）GFP輝度の平均。mask_keys の全マスクを使う"""
    values = []
    for pair in pairs:
        original = imread(pair['original'])
        masks = [imread(pair[key]) for key in mask_keys]
        if original is None or any(m is None for m in masks):
            continue
        gfp = original[:, :, 1]
        for mask_img in masks:
            masked = cv2.bitwise_and(gfp, gfp, mask=combined_color_mask(mask_img, colors))
            values.extend(masked[masked > 0].ravel())
    if not values:
        print("警告: マスク内の輝度が取得できなかったため Global Mean = 1.0 とします。")
        return 1.0
    return float(np.mean(values))


def image_scaling_factor(original, mask_img, global_mean, colors):
    """scaling = Global Mean / この画像のマスク内平均"""
    current = mean_in_mask(original[:, :, 1], combined_color_mask(mask_img, colors))
    if current is None:
        current = global_mean  # 0割防止
    return global_mean / current, current


def extract_rois(original, mask_img, scaling, params, annotate_on=None,
                 side_colors=SIDE_COLORS, overlay_alpha=None):
    """1枚のマスクから左右の正規化済みROIを集める

    annotate_on: アノテーションを描く元の画像（None なら描かない）
    overlay_alpha: 指定すると塗りを半透明で重ねる（モザイク用）

    返り値: ({'left': [...], 'right': [...]}, アノテーション画像 or None)
    """
    gfp = original[:, :, 1]
    center_x = image_center_x(mask_img)
    rois = {side: [] for side in SIDES}

    debug_img = None
    if annotate_on is not None:
        debug_img = annotate_on.copy()
        cv2.line(debug_img, (center_x, 0), (center_x, debug_img.shape[0]), (255, 255, 255), 1)

    for contour in iter_horseshoe_contours(mask_img, params.colors, params.min_area):
        side = classify_side(contour, center_x, params.center_margin)
        if side is None:
            continue

        local_mask, (x, y, w, h) = filled_local_mask(contour)
        if params.dilate_occupancy:
            local_mask, iterations = dilate_to_occupancy(local_mask, params.dilate_occupancy)
            if iterations > params.max_dilate:
                continue

        if debug_img is not None:
            final_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_contours:
                absolute = final_contours[0] + np.array([x, y])
                if overlay_alpha is None:
                    cv2.drawContours(debug_img, [absolute], -1, side_colors[side], -1)
                else:
                    overlay = debug_img.copy()
                    cv2.drawContours(overlay, [absolute], -1, side_colors[side], -1)
                    debug_img = cv2.addWeighted(overlay, overlay_alpha, debug_img, 1 - overlay_alpha, 0)

        roi = gfp[y:y + h, x:x + w]
        roi = cv2.bitwise_and(roi, roi, mask=local_mask)
        resized = cv2.resize(roi, tuple(params.target_size), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float64) * scaling
        if params.clip_uint8:
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        rois[side].append(normalized)

    return rois, debug_img


def collect_image_set(pairs, global_mean, params, mask_key='mask', annotate=True,
                      annotate_key='original', side_colors=SIDE_COLORS, overlay_alpha=None):
    """画像セット全体の左右ROIとアノテーション画像を集める

    明るさの scaling は常に 'mask' のマスクから計算し、ROI は mask_key のマスクから取る。
    """
    rois = {side: [] for side in SIDES}
    debug_images = []
    for pair in pairs:
        original = imread(pair['original'])
        mask_img = imread(pair['mask'])
        target_mask = mask_img if mask_key == 'mask' else imread(pair.get(mask_key))
        base = imread(pair.get(annotate_key)) if annotate else None
        if original is None or mask_img is None or target_mask is None or (annotate and base is None):
            print(f"  画像を読み込めませんでした（スキップ）: {pair['original'].name}")
            continue

        scaling, current = image_scaling_factor(original, mask_img, global_mean, params.colors)
        image_rois, debug_img = extract_rois(original, target_mask, scaling, params,
                                             annotate_on=base, side_colors=side_colors,
                                             overlay_alpha=overlay_alpha)
        for side in SIDES:
            rois[side].extend(image_rois[side])
        if debug_img is not None:
            debug_images.append(debug_img)
        print(f"  {pair['original'].name}: 平均輝度 {current:.2f}, scaling {scaling:.3f}, "
              f"Ventral {len(image_rois['left'])} 個, Dorsal {len(image_rois['right'])} 個")
    return rois, debug_images


def average_heatmap(roi_list, target_size, trim=0, label=''):
    """ROIの画素ごとの平均（trim > 0 なら上下 trim 個ずつ除いたトリム平均）"""
    if not roi_list:
        print(f"警告: {label} の馬蹄形が検出されませんでした（ヒートマップは0）。")
        return np.zeros(tuple(target_size)[::-1], dtype=np.float64)
    stack = np.array(roi_list, dtype=np.float64)
    if trim > 0:
        if len(stack) <= 2 * trim:
            print(f"警告: {label} は {len(stack)} 個しかないため trim={trim} は使わず通常の平均にします。")
        else:
            stack = np.sort(stack, axis=0)[trim:-trim]
    print(f"{label}: {len(roi_list)} 個を平均しました。")
    return stack.mean(axis=0)


def to_uint8_minmax(heatmap):
    """最小値〜最大値を 0〜255 に引き伸ばす（左右それぞれ独立に正規化）"""
    if not np.any(heatmap):
        return np.zeros_like(heatmap, dtype=np.uint8)
    return cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def to_uint8_by_max(heatmap, max_value):
    """共通の最大値で 0〜255 にそろえる（複数のヒートマップを同じスケールで比較する用）"""
    if not max_value:
        max_value = 1.0
    return np.clip(heatmap / max_value * 255.0, 0, 255).astype(np.uint8)
