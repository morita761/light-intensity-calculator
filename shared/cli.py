"""画像セットを使うスクリプト用のコマンドライン補助"""
import argparse
from pathlib import Path

from .horseshoe import load_image_sets


def preload_image_sets(default_path, base_dir):
    """--image-sets を先に読んで画像セットを返す（選択肢を -h に出すため）

    返り値: (txt のパス, {セット名: ImageSet})
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--image-sets', default=str(default_path))
    known, _ = pre.parse_known_args()
    path = Path(known.image_sets)
    return path, load_image_sets(path, base_dir)


def add_image_sets_arg(parser, path):
    parser.add_argument('--image-sets', default=str(path),
                        help='画像セットを定義した txt（既定: config/image_sets.txt）')
