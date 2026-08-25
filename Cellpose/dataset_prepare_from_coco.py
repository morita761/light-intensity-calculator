"""
Detectron2用に作成済みのCOCO形式アノテーション（Detectron2/annotations/*.json）を、
Cellpose のファインチューニングで使える形式（画像 + インスタンスラベル画像）に変換する。

新規にアノテーションを取り直す必要はなく、既存の馬蹄形アノテーションをそのまま再利用できる。

Cellposeの学習規約:
    001.png
    001_masks.tif   ... 0=背景, 1..N=インスタンスID の2Dラベル画像（uint16）

使い方:
    python dataset_prepare_from_coco.py \
        --coco ../Detectron2/annotations/train_annotations_horseshoe.json \
        --images-dir ../data/train/images \
        --out-dir ./cellpose_dataset/train

    python dataset_prepare_from_coco.py \
        --coco ../Detectron2/annotations/val_annotations_horseshoe.json \
        --images-dir ../data/val/images \
        --out-dir ./cellpose_dataset/test
"""
import argparse
import json
import os

import cv2
import numpy as np
import tifffile

from func.cli_errors import run_main


def build_instance_mask(height, width, annotations, category_id):
    """
    1画像分のCOCOアノテーション(polygon)から、インスタンスラベル画像(uint16)を作る。
    horseshoe以外のカテゴリ(negative等)は背景として扱う。
    """
    label_mask = np.zeros((height, width), dtype=np.uint16)
    instance_id = 1
    for ann in annotations:
        if ann["category_id"] != category_id:
            continue
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(label_mask, [pts], color=instance_id)
        instance_id += 1
    return label_mask


def main():
    parser = argparse.ArgumentParser(description="COCOアノテーション -> Cellpose学習データ変換")
    parser.add_argument("--coco", required=True, help="Detectron2形式のCOCO JSONパス")
    parser.add_argument("--images-dir", required=True, help="元画像が置かれているディレクトリ")
    parser.add_argument("--out-dir", required=True, help="変換後データの出力先ディレクトリ")
    parser.add_argument(
        "--category-name", default="horseshoe", help="ラベル化する対象カテゴリ名（デフォルト: horseshoe）"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        raise NotADirectoryError(f"--images-dir が見つかりません: {args.images_dir}")

    with open(args.coco, "r") as f:
        coco = json.load(f)

    category_id = None
    for cat in coco["categories"]:
        if cat["name"] == args.category_name:
            category_id = cat["id"]
            break
    if category_id is None:
        raise ValueError(f"カテゴリ '{args.category_name}' が見つかりません: {coco['categories']}")

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(args.out_dir, exist_ok=True)

    n_written = 0
    for img_info in coco["images"]:
        src_path = os.path.join(args.images_dir, img_info["file_name"])
        if not os.path.exists(src_path):
            print(f"[スキップ] 画像が見つかりません: {src_path}")
            continue

        image = cv2.imread(src_path)
        if image is None:
            print(f"[スキップ] 画像を読み込めません: {src_path}")
            continue

        anns = anns_by_image.get(img_info["id"], [])
        label_mask = build_instance_mask(img_info["height"], img_info["width"], anns, category_id)
        n_instances = int(label_mask.max())

        basename = os.path.splitext(img_info["file_name"])[0]
        out_image_path = os.path.join(args.out_dir, f"{basename}.png")
        out_mask_path = os.path.join(args.out_dir, f"{basename}_masks.tif")

        cv2.imwrite(out_image_path, image)
        tifffile.imwrite(out_mask_path, label_mask)

        print(f"[変換] {img_info['file_name']}: {n_instances} インスタンス -> {out_mask_path}")
        n_written += 1

    print(f"完了: {n_written} 画像を {args.out_dir} に出力しました。")


if __name__ == "__main__":
    run_main(main)
