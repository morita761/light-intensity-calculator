"""
COCOアノテーション(Detectron2/annotations/*.json)から、対象カテゴリ(デフォルト: horseshoe)の
平均インスタンスサイズを計算し、Cellpose推論時に指定すべき --diameter の目安値を出力する。

README.md「検証結果」で確認した通り、Cellposeのゼロショット検出は diameter を
実際の対象サイズに合わせないと、組織テクスチャの別スケール（もっと粗い/細かい単位）を
拾ってしまう。この値を毎回目視・手計算する代わりに、アノテーションから機械的に算出する。

diameterはCellposeの定義（インスタンス面積から求める等価直径: 2*sqrt(area/pi)）に合わせて計算する。

使い方:
    python estimate_diameter.py --coco ../Detectron2/annotations/train_annotations_horseshoe.json
"""
import argparse
import json
import math

from func.cli_errors import run_main


def polygon_area(seg):
    """
    COCOのsegmentation(polygon, [x1,y1,x2,y2,...])からShoelace公式で面積を計算する。
    """
    xs = seg[0::2]
    ys = seg[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def instance_area(ann):
    if "area" in ann and ann["area"]:
        return float(ann["area"])
    return sum(polygon_area(seg) for seg in ann["segmentation"])


def main():
    parser = argparse.ArgumentParser(description="COCOアノテーションから推奨diameterを算出")
    parser.add_argument("--coco", required=True, help="Detectron2形式のCOCO JSONパス")
    parser.add_argument("--category-name", default="horseshoe", help="対象カテゴリ名")
    args = parser.parse_args()

    with open(args.coco, "r") as f:
        coco = json.load(f)

    category_id = None
    for cat in coco["categories"]:
        if cat["name"] == args.category_name:
            category_id = cat["id"]
            break
    if category_id is None:
        raise ValueError(f"カテゴリ '{args.category_name}' が見つかりません: {coco['categories']}")

    diameters = []
    bbox_w, bbox_h = [], []
    for ann in coco["annotations"]:
        if ann["category_id"] != category_id:
            continue
        area = instance_area(ann)
        diameters.append(2.0 * math.sqrt(area / math.pi))
        if "bbox" in ann:
            _, _, w, h = ann["bbox"]
            bbox_w.append(w)
            bbox_h.append(h)

    if not diameters:
        raise ValueError(f"カテゴリ '{args.category_name}' のインスタンスが見つかりません")

    mean_diameter = sum(diameters) / len(diameters)
    print(f"対象: {args.coco} (category='{args.category_name}')")
    print(f"インスタンス数: {len(diameters)}")
    print(f"等価直径(2*sqrt(area/pi)) 平均: {mean_diameter:.1f}px  (min={min(diameters):.1f}, max={max(diameters):.1f})")
    if bbox_w:
        print(f"bbox幅 平均: {sum(bbox_w) / len(bbox_w):.1f}px, bbox高さ 平均: {sum(bbox_h) / len(bbox_h):.1f}px")
    print(f"\n推奨: python inference_test.py --diameter {mean_diameter:.0f} ...")


if __name__ == "__main__":
    run_main(main)
