"""
ND2形式のZ-stack画像に対し、Z方向の複数範囲で馬蹄形検出を行う改善版パイプライン。

背景:
    1枚のZスライスだけで検出すると、Z方向のノイズや、異なるZ位置にある軸索が
    重なって見えることによる誤検出が起きやすい。そこで、1画像あたり約20枚ある
    Zスライスを複数のZ範囲に分けてそれぞれ重ね合わせ(投影)を作り、範囲ごとに
    独立して馬蹄形検出を行う。各範囲の結果は統合せず、後から比較してどのZ範囲が
    最も適切に検出できているかを評価できるようにする。

検出する範囲（デフォルト、n_z=20の場合）:
    1. 全Z-stackをまとめた投影 (Z1-20)
    2. 5枚ずつ、3枚おきにずらした投影 (Z1-5, Z4-8, Z7-11, Z10-14, Z13-17, Z16-20)

出力（--out-dir 以下、範囲ごとにサブディレクトリを作成し、統合しない）:
    <range_label>/projection.png   投影画像（正規化後のグレースケール）
    <range_label>/overlay.png      検出結果のオーバーレイ画像
    <range_label>/masks.tif        インスタンスラベル画像（後で読み直せる）
    <range_label>/instances.csv    インスタンスごとの重心・面積・V/D・平均輝度
    summary.csv                    範囲ごとの検出数・平均輝度などの比較表
    comparison_montage.png         全範囲のoverlayを並べた比較用画像

使い方:
    python inference_nd2_zstack.py --nd2 path/to/image.nd2
    python inference_nd2_zstack.py --nd2 path/to/image.nd2 --diameter 40 --gpu
    python inference_nd2_zstack.py --nd2 path/to/image.nd2 --window-size 5 --step 3 --no-full-stack
"""
import argparse
import csv
import os

import cv2
import numpy as np
import tifffile

from func.cellpose_model import eval_image, load_model
from func.cli_errors import require_file, run_main
from func.vd_split import (
    green_center_x,
    masks_to_instances,
    split_left_right,
    summarize_intensity,
)
from func.zstack import load_nd2_zstack, project, to_uint8, zstack_ranges

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "zstack")


def process_range(model, stack, label, z_indices, args, out_dir):
    range_dir = os.path.join(out_dir, label)
    os.makedirs(range_dir, exist_ok=True)

    projected = project(stack, z_indices, method=args.projection)
    projected_u8 = to_uint8(projected)
    cv2.imwrite(os.path.join(range_dir, "projection.png"), projected_u8)

    image = cv2.cvtColor(projected_u8, cv2.COLOR_GRAY2BGR)
    masks = eval_image(
        model,
        image,
        args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )
    tifffile.imwrite(os.path.join(range_dir, "masks.tif"), masks.astype(np.uint16))

    instances = masks_to_instances(masks, min_area=args.min_area)
    center_x = green_center_x(image)
    left, right = split_left_right(instances, center_x)
    left_vals, left_mean = summarize_intensity(image, left)
    right_vals, right_mean = summarize_intensity(image, right)

    with open(os.path.join(range_dir, "instances.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "side", "centroid_x", "centroid_y", "area", "mean_intensity"])
        for inst, val in zip(left, left_vals):
            writer.writerow([inst["label_id"], "V(left)", *inst["centroid"], inst["area"], val])
        for inst, val in zip(right, right_vals):
            writer.writerow([inst["label_id"], "D(right)", *inst["centroid"], inst["area"], val])

    overlay_path = os.path.join(range_dir, "overlay.png")
    save_overlay(image, masks, left, right, label, overlay_path)

    return {
        "range_label": label,
        "z_start": z_indices[0] + 1,
        "z_end": z_indices[-1] + 1,
        "n_slices": len(z_indices),
        "n_instances_left": len(left),
        "n_instances_right": len(right),
        "n_instances_total": len(left) + len(right),
        "mean_intensity_left": left_mean,
        "mean_intensity_right": right_mean,
        "overlay_path": overlay_path,
    }


def save_overlay(image, masks, left, right, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(6, 6))
    n = int(masks.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.random((max(n, 1), 3))
    colors[0] = 0
    overlay = np.ma.masked_where(masks == 0, masks)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.imshow(overlay, cmap=ListedColormap(colors), alpha=0.6)
    for inst in left:
        cx, cy = inst["centroid"]
        ax.plot(cx, cy, "o", color="cyan", markersize=3)
    for inst in right:
        cx, cy = inst["centroid"]
        ax.plot(cx, cy, "o", color="magenta", markersize=3)
    ax.set_title(f"{title} (n={len(left) + len(right)}) cyan=V magenta=D")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_montage(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.reshape(-1)
    for ax, r in zip(axes, results):
        ax.imshow(mpimg.imread(r["overlay_path"]))
        ax.set_title(f"{r['range_label']} (n={r['n_instances_total']})", fontsize=10)
        ax.axis("off")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[比較画像] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ND2形式のZ-stack画像に対し、Z方向の複数範囲で馬蹄形検出を行う（各範囲は独立保存・後で比較評価する）"
    )
    parser.add_argument("--nd2", required=True, help="入力ND2ファイルのパス")
    parser.add_argument(
        "--channel", type=int, default=None, help="複数チャンネルがある場合に使用するチャンネル番号（省略時は先頭）"
    )
    parser.add_argument("--window-size", type=int, default=5, help="スライディング範囲のZスライス枚数")
    parser.add_argument("--step", type=int, default=3, help="スライディング範囲をずらすZスライス数")
    parser.add_argument(
        "--no-full-stack", action="store_true", help="全Z-stackをまとめた投影による検出をスキップする"
    )
    parser.add_argument(
        "--projection", choices=["max", "mean", "sum"], default="max", help="Z方向の重ね合わせ方式"
    )
    parser.add_argument("--diameter", type=float, default=None, help="想定オブジェクト直径(px)。Noneなら自動推定")
    parser.add_argument("--gpu", action="store_true", help="GPUを使用する")
    parser.add_argument("--min-area", type=int, default=20, help="有効インスタンスとみなす最小面積(px^2)")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="fine-tuning済みモデルのパス（省略時は事前学習済みのcpsam_v2をゼロショットで使用）",
    )
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument(
        "--out-dir", type=str, default=None, help="出力先ディレクトリ（省略時は ./output/zstack/<ND2ファイル名>）"
    )
    args = parser.parse_args()

    require_file(args.nd2, "入力ND2ファイル(--nd2)")

    stack = load_nd2_zstack(args.nd2, channel=args.channel)
    n_z = stack.shape[0]
    print(f"[Z-stack読込] {args.nd2}  shape={stack.shape} (Z={n_z})")

    ranges = zstack_ranges(
        n_z, window_size=args.window_size, step=args.step, include_full=not args.no_full_stack
    )
    print(f"[Z範囲] {len(ranges)}件: {[label for label, _ in ranges]}")

    basename = os.path.splitext(os.path.basename(args.nd2))[0]
    out_dir = args.out_dir or os.path.join(OUTPUT_ROOT, basename)
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(use_gpu=args.gpu, pretrained_model=args.pretrained_model)

    results = []
    for label, z_indices in ranges:
        print(f"\n=== Z範囲: {label} ({len(z_indices)}枚) ===")
        result = process_range(model, stack, label, z_indices, args, out_dir)
        results.append(result)
        print(
            f"  V(左)={result['n_instances_left']}個(平均輝度{result['mean_intensity_left']:.2f}), "
            f"D(右)={result['n_instances_right']}個(平均輝度{result['mean_intensity_right']:.2f})"
        )

    summary_path = os.path.join(out_dir, "summary.csv")
    fieldnames = [
        "range_label",
        "z_start",
        "z_end",
        "n_slices",
        "n_instances_left",
        "n_instances_right",
        "n_instances_total",
        "mean_intensity_left",
        "mean_intensity_right",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[比較表] {summary_path}")

    save_montage(results, os.path.join(out_dir, "comparison_montage.png"))

    print(f"\n完了: {len(results)}個のZ範囲の検出結果を統合せず、それぞれ独立して {out_dir} へ保存しました。")
    print("summary.csv と comparison_montage.png で各範囲の検出結果を比較してください。")


if __name__ == "__main__":
    run_main(main)
