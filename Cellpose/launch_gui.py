"""
Cellpose GUI を起動し、human-in-the-loop（手動修正 + fine-tuning）を行うためのランチャー。

train_finetune.py（CLIでの一括fine-tuning）とは別に、Cellpose公式GUIでは
「ゼロショット推論 → マスクを手動修正 → その場でfine-tuning」のループを
インタラクティブに回せる。実行・精度検証はGUI操作が必要なため人手で行う想定で、
本スクリプトはその起動と事前準備だけを整える。

事前準備（推奨）:
    dataset_prepare_from_coco.py で既存のCOCOアノテーションを変換しておくと、
    画像ごとに `<画像名>_masks.tif` が生成される。GUI起動後に
    File > "Autoload masks from _masks.tif file" を有効にしておけば、
    画像を開いた時点で変換済みのマスクがそのまま読み込まれ、
    ゼロから手動アノテーションし直す必要がない。

    python dataset_prepare_from_coco.py \
        --coco ../Detectron2/annotations/train_annotations_horseshoe.json \
        --images-dir ../data/train/images \
        --out-dir ./cellpose_dataset/train

使い方:
    # インストール（初回のみ）
    pip install "cellpose[gui]"

    # GUI起動（指定した画像をプリロード。省略時は空の状態で起動）
    python launch_gui.py --image ./cellpose_dataset/train/001.png

GUI内での操作手順（このスクリプトは起動のみ。以降は手動）:
    1. File > "Load image" で対象画像を開く（--image指定時は自動で開かれる）
    2. File > "Autoload masks from _masks.tif file" にチェックを入れておくと、
       同名の `_masks.tif`（本リポジトリでは dataset_prepare_from_coco.py の出力）
       が自動読み込みされる。無ければ現行の学習済みモデルで推論して初期マスクを作る。
    3. 誤検出・見逃しをブラシ/右クリック削除で手動修正し、Ctrl+S で保存
       （`<画像名>_seg.npy` として保存される）
    4. 同じフォルダ内の複数画像で 1-3 を繰り返す
    5. Models > "Train new model with image+masks in folder" を実行
       （ダイアログで learning_rate / n_epochs / model_name を設定）
    6. 学習後のモデルは `<フォルダ>/models/<model_name>` に保存され、
       GUIのモデル一覧にも自動追加されて他画像へすぐ試せる
    7. 保存されたモデルは inference_test.py --pretrained-model <パス> で
       CLIからも利用できる
"""
import argparse

from cellpose.gui import gui


def main():
    parser = argparse.ArgumentParser(description="Cellpose GUI 起動")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="起動時にプリロードする画像パス（省略可。GUI内から後で開いてもよい）",
    )
    args = parser.parse_args()

    print("Cellpose GUIを起動します。")
    print("手順は launch_gui.py 冒頭のdocstringを参照してください。")
    if args.image:
        print(f"プリロード画像: {args.image}")

    gui.run(image=args.image)


if __name__ == "__main__":
    main()
