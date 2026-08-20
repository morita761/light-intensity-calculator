"""
既存の馬蹄形アノテーション（dataset_prepare_from_coco.py で変換済み）を使って、
Cellpose-SAM の事前学習モデル(cpsam_v2)を馬蹄形検出用にファインチューニングするスクリプト。

ゼロショットの cpsam_v2 は「馬蹄形と似た大きさ・形状の構造」自体は高精度に検出できるが、
このプロジェクトの馬蹄形は組織テクスチャ中の特定条件を満たすサブセットであるため、
ゼロショットのままでは背景のテクスチャも大量に拾ってしまう（README.md 参照）。
このスクリプトで少数の正解データを使って再学習し、"horseshoe" クラスだけを
選択的に検出できるようにする。

事前準備:
    python dataset_prepare_from_coco.py --coco ../Detectron2/annotations/train_annotations_horseshoe.json \
        --images-dir ../data/train/images --out-dir ./cellpose_dataset/train
    python dataset_prepare_from_coco.py --coco ../Detectron2/annotations/val_annotations_horseshoe.json \
        --images-dir ../data/val/images --out-dir ./cellpose_dataset/test

使い方:
    python train_finetune.py --train-dir ./cellpose_dataset/train --test-dir ./cellpose_dataset/test
"""
import argparse
import os

from cellpose import io as cp_io
from cellpose import models, train


def main():
    parser = argparse.ArgumentParser(description="Cellpose ファインチューニング（馬蹄形検出）")
    parser.add_argument("--train-dir", required=True, help="dataset_prepare_from_coco.py の出力先(train)")
    parser.add_argument("--test-dir", default=None, help="dataset_prepare_from_coco.py の出力先(val)")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-name", default="horseshoe_cpsam")
    parser.add_argument("--save-path", default="./output")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print("[データ読込]")
    output = cp_io.load_train_test_data(
        args.train_dir, args.test_dir, mask_filter="_masks"
    )
    train_data, train_labels, _, test_data, test_labels, _ = output
    print(f"  train: {len(train_data)} 枚, test: {len(test_data) if test_data else 0} 枚")

    print("[事前学習モデル読込] cpsam_v2")
    model = models.CellposeModel(gpu=args.gpu)

    print("[ファインチューニング開始]")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model_name=args.model_name,
        save_path=args.save_path,
    )

    print(f"[完了] モデル保存先: {model_path}")
    print("推論時は models.CellposeModel(pretrained_model=<model_path>) で読み込めます。")


if __name__ == "__main__":
    main()
