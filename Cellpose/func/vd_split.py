"""
Cellpose の推論結果（インスタンスラベル画像）から
Detectron2パイプラインと同様の V-D（Ventral/Dorsal）左右分割・輝度計算を行う共通関数。

Detectron2/func/intensityCalc.py, Detectron2/func/splitobje.py の
HSV緑色抽出・左右分割ロジックを踏襲している。
"""
import cv2
import numpy as np


def green_center_x(image_bgr):
    """
    画像内の緑色領域のバウンディングボックス中心のX座標を返す。
    Detectron2/func/splitobje.py の split_left_right() と同じロジック。
    """
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(green_mask) == 0:
        return width // 2

    x, y, w, h = cv2.boundingRect(cv2.findNonZero(green_mask))
    return x + w // 2


def masks_to_instances(masks, min_area=20, max_area=None):
    """
    cellpose の eval() が返すラベル画像 (H, W, int) を、
    インスタンスごとの (label_id, bool_mask, centroid_xy, area) のリストに変換する。
    """
    instances = []
    for label_id in range(1, int(masks.max()) + 1):
        instance_mask = masks == label_id
        area = int(instance_mask.sum())
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        ys, xs = np.where(instance_mask)
        centroid = (float(xs.mean()), float(ys.mean()))
        instances.append({
            "label_id": label_id,
            "mask": instance_mask,
            "centroid": centroid,
            "area": area,
        })
    return instances


def split_left_right(instances, center_x):
    """
    重心のX座標を基準にインスタンスを左右(V/D)に分割する。
    """
    left = [inst for inst in instances if inst["centroid"][0] < center_x]
    right = [inst for inst in instances if inst["centroid"][0] >= center_x]
    return left, right


def mean_intensity(image_bgr, instance_mask, channel=1):
    """
    指定チャンネル（デフォルトはGチャンネル）のマスク内平均輝度を返す。
    """
    channel_img = image_bgr[:, :, channel]
    values = channel_img[instance_mask]
    return float(values.mean()) if values.size > 0 else 0.0


def summarize_intensity(image_bgr, instances, channel=1):
    """
    インスタンスのリストから平均輝度のリストと全体平均を計算する。
    """
    values = [mean_intensity(image_bgr, inst["mask"], channel) for inst in instances]
    overall = float(np.mean(values)) if values else 0.0
    return values, overall
