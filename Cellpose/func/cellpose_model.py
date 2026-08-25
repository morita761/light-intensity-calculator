"""
Cellposeモデルのロード・推論を行う共通処理。

画像パスから読み込む場合(inference_test.py)、Z-stack投影画像のように
既にメモリ上にある配列から推論する場合(inference_nd2_zstack.py)の
両方から共通で使う。
"""
from cellpose import models


def load_model(use_gpu=False, pretrained_model=None):
    if pretrained_model:
        print(f"[モデル準備] CellposeModel (pretrained_model='{pretrained_model}') ※fine-tuning済みモデル")
        return models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)
    print("[モデル準備] CellposeModel (pretrained_model='cpsam_v2') ※初回はモデルを自動ダウンロードします")
    return models.CellposeModel(gpu=use_gpu)


def eval_image(model, image, diameter, flow_threshold=0.4, cellprob_threshold=0.0):
    print(
        f"[推論実行] diameter={diameter}, flow_threshold={flow_threshold}, "
        f"cellprob_threshold={cellprob_threshold}"
    )
    masks, flows, styles = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    n_instances = int(masks.max())
    print(f"  検出インスタンス数: {n_instances}")
    return masks
