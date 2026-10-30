"""画像の明度ヒストグラムと、CLAHE（局所適応ヒストグラム平坦化）後のヒストグラムを比較

用途: グレースケール化した画像と、CLAHE 適用後の画像を、それぞれの明度ヒストグラムと
      並べて表示する（2行2列）。前処理でコントラストがどう変わるかの確認用。
      参考: https://www.codevace.com/py-opencv-clahe/
入力: data/up_Fz_green_stronger.tif（--image で変更可）
出力: output/clahe_histogram/<画像名>_clahe.png

Usage:
    python clahe_histogram.py
    python clahe_histogram.py --image other.tif --clip-limit 3.0 --tile 16
    python clahe_histogram.py --both --no-show
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from lib.paths import DATA_DIR, FIGURE_SETTINGS_FILE, OUTPUT_DIR
from shared.figure import finish, make_parser, save_figure
from shared.horseshoe import imread
from shared.settings import load_figure_settings

NAME = 'clahe_histogram'


def main():
    s = load_figure_settings(FIGURE_SETTINGS_FILE, NAME)
    parser = make_parser(__doc__, NAME)
    parser.add_argument('--image', default=str(DATA_DIR / s.str('image', 'up_Fz_green_stronger.tif')),
                        help='入力画像のパス')
    parser.add_argument('--clip-limit', type=float, default=s.float('clip_limit', 2.0), help='CLAHE の clipLimit')
    parser.add_argument('--tile', type=int, default=s.int('tile', 8), help='CLAHE のタイル数（tile x tile）')
    args = parser.parse_args()

    path = Path(args.image)
    image = imread(path)
    if image is None:
        raise SystemExit(f"エラー: 画像を読み込めませんでした: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 環境に依存しないようにここでグレースケール化
    clahe = cv2.createCLAHE(clipLimit=args.clip_limit, tileGridSize=(args.tile, args.tile))
    result = clahe.apply(gray)
    print(f"{path.name}: clipLimit={args.clip_limit}, tileGridSize=({args.tile}, {args.tile})")

    fig, axes = plt.subplots(2, 2, figsize=s.floats('figsize', (12, 7.5)))
    for row, (img, prefix) in enumerate(((gray, 'original_'), (result, 'clahe_'))):
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].axis('off')
        axes[row, 0].set_title(s.str(prefix + 'title'), fontsize=s.float('title_fontsize'))
        axes[row, 1].plot(cv2.calcHist([img], [0], None, [256], [0, 256]))
        s.apply(axes[row, 1], prefix=('hist_' + prefix, 'hist_'))
    fig.tight_layout()
    save_figure(fig, f'{path.stem}_clahe', args, s, OUTPUT_DIR / NAME)
    finish(args)


if __name__ == '__main__':
    main()
