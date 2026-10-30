"""plot_gradient.py の動作確認用ダミー画像を作る

用途: 緑色のグラディエント（非線形・線形など3種類）を青い長方形で囲んだ画像を作る。
      元画像とマスクを兼ねるので、image_sets.txt の [dummy] でそのまま使える。
      旧 dumy-plot-gradient.py のダミー画像生成部分。
出力: data/sample/dummy_multi_gradient_image.png

Usage:
    python make_dummy_gradient.py
    python plot_gradient.py --set dummy
"""
import argparse

import cv2
import numpy as np

from lib.paths import SAMPLE_DIR
from shared.horseshoe import imwrite

# (x, y, 幅, 高さ, 指数): 輝度 = 255 * (位置 / 幅) ** 指数
REGIONS = [(50, 50, 300, 100, 1.5), (100, 200, 305, 120, 0.8), (200, 400, 300, 90, 1.0)]


def generate(width=800, height=600):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, w, h, power in REGIONS:
        for i in range(w):
            img[y:y + h, x + i, 1] = int(255 * (i / w) ** power)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return img


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    path = SAMPLE_DIR / 'dummy_multi_gradient_image.png'
    imwrite(path, generate())
    print(f"保存: {path}")
