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

学習の監視（loss / learning rate）:
    学習完了後、以下を --save-path 以下に出力する。
        <model_name>_loss_history.csv   epochごとのtrain/test loss
        <model_name>_loss_curve.png     loss曲線のプロット
        tensorboard/<model_name>/       TensorBoard用ログ（Detectron2と同じ見方でtotal_lossを確認可能）
    TensorBoardでの確認:
        tensorboard --logdir <save-path>/tensorboard
    Cellposeのtrain_seg()はlearning_rateを固定値（またはcosine decay、cellpose内部実装依存）として
    扱うため、動的な学習率スケジュールそのものは公開されていない。ここでは実行時に指定した
    --learning-rate の値をTensorBoardに定数として記録し、どの学習率で学習したログかを追跡できるようにする。
"""
import argparse
import csv
import os

from cellpose import io as cp_io
from cellpose import models, train

from func.cli_errors import run_main


def save_loss_history(save_path, model_name, train_losses, test_losses, learning_rate):
    """
    epochごとのtrain/test lossをCSV保存し、TensorBoardにも記録する。
    Detectron2/tensorboard/README.md と同じ「total_lossが下がっているか/収束したか/
    振動していないか」という観点でCellposeの学習も監視できるようにする。
    """
    n_epochs = len(train_losses)
    csv_path = os.path.join(save_path, f"{model_name}_loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])
        for epoch in range(n_epochs):
            test_loss = test_losses[epoch] if test_losses is not None and epoch < len(test_losses) else ""
            writer.writerow([epoch, train_losses[epoch], test_loss])
    print(f"[loss履歴] {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(n_epochs), train_losses, label="train_loss")
        if test_losses is not None and len(test_losses) > 0:
            ax.plot(range(len(test_losses)), test_losses, label="test_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{model_name} loss curve (learning_rate={learning_rate})")
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(save_path, f"{model_name}_loss_curve.png")
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[loss曲線] {png_path}")
    except ImportError:
        print("[警告] matplotlibが無いためloss曲線のプロットをスキップしました")

    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(save_path, "tensorboard", model_name)
        writer = SummaryWriter(log_dir=log_dir)
        for epoch in range(n_epochs):
            writer.add_scalar("loss/train", train_losses[epoch], epoch)
            if test_losses is not None and epoch < len(test_losses):
                writer.add_scalar("loss/test", test_losses[epoch], epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)
        writer.close()
        print(f"[TensorBoard] {log_dir}  (確認: tensorboard --logdir {os.path.join(save_path, 'tensorboard')})")
    except ImportError:
        print("[警告] tensorboardが無いためTensorBoardログの出力をスキップしました（pip install tensorboard）")


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

    if not os.path.isdir(args.train_dir):
        raise NotADirectoryError(f"--train-dir が見つかりません: {args.train_dir}")
    if args.test_dir is not None and not os.path.isdir(args.test_dir):
        raise NotADirectoryError(f"--test-dir が見つかりません: {args.test_dir}")

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

    save_loss_history(args.save_path, args.model_name, train_losses, test_losses, args.learning_rate)

    print(f"[完了] モデル保存先: {model_path}")
    print("推論時は models.CellposeModel(pretrained_model=<model_path>) で読み込めます。")


if __name__ == "__main__":
    run_main(main)
