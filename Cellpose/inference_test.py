"""
Cellpose (Cellpose-SAM) の事前学習済みモデルをゼロショットで動かし、
馬蹄形（horseshoe）らしき構造をどの程度検出できるか確認するテストスクリプト。

学習は一切行わず、公式の汎用モデル（cpsam_v2）をそのまま推論に使う。
検出結果は ./output/ 以下にオーバーレイ画像として保存される。

使い方:
    python inference_test.py
    python inference_test.py --image ../pic/up_Fz_green_stronger_selected_2_blue.tif --diameter 40
    python inference_test.py --gpu

参考（動作確認済み）:
    このスクリプトは cellpose==4.2.1.1 (CPU) で実際に実行し、
    - Detectron2/simpledataset/images/003.png（合成の馬蹄形アイコン画像）では
      重なり合う凹形状も含めてほぼ正しくインスタンス分割できることを確認済み。
    - pic/ 以下の実データ(TIFF)では、diameter を実際の馬蹄形サイズ
      （COCOアノテーションの平均: 幅約49px, 高さ約35px）に合わせないと、
      組織のテクスチャ単位（もっと粗い/細かいスケール）を拾ってしまう。
    詳細は README.md の「検証結果」を参照。
"""
import argparse
import glob
import os

import cv2
import numpy as np
from cellpose import io as cp_io
from cellpose import models

from func.vd_split import (
    green_center_x,
    masks_to_instances,
    split_left_right,
    summarize_intensity,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def find_default_images():
    """
    実データ (pic/ は .gitignore 対象なのでクローン直後は存在しない場合がある) を優先し、
    無ければリポジトリに同梱されている合成テスト画像にフォールバックする。
    """
    real = sorted(glob.glob(os.path.join(REPO_ROOT, "pic", "*.tif")))
    if real:
        return real[:1]
    synthetic = sorted(
        glob.glob(os.path.join(REPO_ROOT, "Detectron2", "simpledataset", "images", "*.png"))
    )
    return synthetic[:1]


def run_cellpose(image_path, diameter, use_gpu):
    print(f"[画像読込] {image_path}")
    image = cp_io.imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(f"  shape={image.shape}, dtype={image.dtype}")

    print("[モデル準備] CellposeModel (pretrained_model='cpsam_v2') ※初回はモデルを自動ダウンロードします")
    model = models.CellposeModel(gpu=use_gpu)

    print(f"[推論実行] diameter={diameter}")
    masks, flows, styles = model.eval(image, diameter=diameter)
    n_instances = int(masks.max())
    print(f"  検出インスタンス数: {n_instances}")

    return image, masks


def visualize(image, masks, instances, left, right, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image)
    axes[0].set_title("input image")
    axes[0].axis("off")

    n = int(masks.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.random((max(n, 1), 3))
    colors[0] = 0
    overlay = np.ma.masked_where(masks == 0, masks)

    axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image)
    axes[1].imshow(overlay, cmap=ListedColormap(colors), alpha=0.6)
    for inst in left:
        cx, cy = inst["centroid"]
        axes[1].plot(cx, cy, "o", color="cyan", markersize=3)
    for inst in right:
        cx, cy = inst["centroid"]
        axes[1].plot(cx, cy, "o", color="magenta", markersize=3)
    axes[1].set_title(f"cellpose masks (n={n - 1}) cyan=left(V) magenta=right(D)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[保存] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cellpose 事前学習モデルによる馬蹄形検出テスト")
    parser.add_argument("--image", type=str, default=None, help="入力画像パス（省略時は自動選択）")
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="想定オブジェクト直径(px)。Noneなら自動推定。実データではおよそ40前後を推奨。",
    )
    parser.add_argument("--gpu", action="store_true", help="GPUを使用する")
    parser.add_argument("--min-area", type=int, default=20, help="有効インスタンスとみなす最小面積(px^2)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_path = args.image
    if image_path is None:
        candidates = find_default_images()
        if not candidates:
            raise FileNotFoundError(
                "テスト用画像が見つかりません。--image で画像パスを指定してください。"
            )
        image_path = candidates[0]

    image, masks = run_cellpose(image_path, args.diameter, args.gpu)

    instances = masks_to_instances(masks, min_area=args.min_area)
    center_x = green_center_x(image)
    left, right = split_left_right(instances, center_x)
    left_vals, left_mean = summarize_intensity(image, left)
    right_vals, right_mean = summarize_intensity(image, right)

    print(f"[V-D分割] center_x={center_x}")
    print(f"  左側(V) インスタンス数: {len(left)}, 平均輝度: {left_mean:.2f}")
    print(f"  右側(D) インスタンス数: {len(right)}, 平均輝度: {right_mean:.2f}")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_cellpose_test.png")
    visualize(image, masks, instances, left, right, out_path)


if __name__ == "__main__":
    main()
