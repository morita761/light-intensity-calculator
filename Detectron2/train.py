from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator # 追加
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2 # 画像表示のため
import numpy as np
import torch
import os

# データセット登録

# register_coco_instances("horseshoe_dataset", {}, "train_annotations.json", "simpledataset/images/")
register_coco_instances("horseshoe_dataset_train", {}, "train_annotations_horseshoe_RGB.json", "../data/2img/")
register_coco_instances("horseshoe_dataset_val", {}, "val_annotations_horseshoe_RGB.json", "../data/2img_val/") # 追加

cfg = get_cfg()
# Mask R-CNN R50-FPN
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# https://communicity-docs.readthedocs.io/en/latest/src/communicity_toolbox/toolbox/Models/detectron2/README.html
cfg.MODEL.WEIGHTS = "../data/model/model_final_f10217.pkl" # ダウンロードしたファイルの相対パス

# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f1027.pkl"

# Mask R-CNN R101-FPN (3x schedule). ResNet-50の代わりに、より深いResNet-101をバックボーンに使用したモデル
# cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"

# cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"

cfg.DATASETS.TRAIN = ("horseshoe_dataset_train",)
cfg.DATASETS.TEST = ("horseshoe_dataset_val",) # ここを変更
# cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.OUTPUT_DIR = "./output"

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
# cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.MAX_ITER = 270000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # horseshoe, negative
cfg.TEST.EVAL_PERIOD = 50 # 50イテレーションごとに評価を実行（任意、デフォルトは0）

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

def custom_mapper(dataset_dict):
    import copy
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # AugmentationListにしない
    augmentations = [
        T.RandomRotation(angle=[-20, 20]),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        # ✅ 追加: 明るさのランダム変動
        T.RandomBrightness(intensity_range=[0.8, 1.2]),
        # ✅ 追加: コントラストのランダム変動
        T.RandomContrast(intensity_range=[0.85, 1.15]),
        T.ResizeShortestEdge([640, 800], max_size=1333),
    ]

    image, transforms = T.apply_transform_gens(augmentations, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

    # 1. 元の "annotations" リストを取得
    original_annos = dataset_dict["annotations"]

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        # for obj in dataset_dict.pop("annotations")
        for obj in original_annos
    ]
    # 3. 🚨 【重要】 変換後のリストで "annotations" キーを上書きする
    # draw_dataset_dict が期待する形式を維持し、かつ座標が画像サイズと一致するようにする
    dataset_dict["annotations"] = annos 

    # 4. Instancesオブジェクトも作成する（これは訓練に必要な形式）
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
    return dataset_dict



# --- カスタムトレーナーの定義 ---
# DefaultTrainerを継承し、カスタムデータセット用の評価器を構築します。
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        指定されたデータセット名に基づいて適切な評価器を構築します。
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True) # 評価結果フォルダも作成

        # "horseshoe_dataset_val" の評価には COCOEvaluator を使用します。
        # データセットがCOCO形式のjsonアノテーションを持っていることを前提としています。
        if dataset_name == "horseshoe_dataset_val":
            print("data set")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
        # 他のデータセット名に対する評価器が必要な場合は、ここに追加できます。
        # 例: elif dataset_name == "another_dataset": ...

        # 未知のデータセット名が指定された場合はエラーを発生させます。
        raise NotImplementedError(
            f"No evaluator implemented for dataset: {dataset_name}. "
            f"Please ensure '{dataset_name}' is correctly handled in CustomTrainer.build_evaluator."
        )

    @staticmethod
    def debug_data_augmentation(cfg):
        """
        カスタムマッパーによるデータ拡張の結果を視覚的に確認するためのヘルパー関数。
        """
        # custom_mapperが適用されたデータローダーを構築
        data_loader = build_detection_train_loader(cfg, mapper=custom_mapper)

        # 最初のバッチを取得
        # data_loaderはイテレータとして動作するため、next()で1バッチ分を取得
        first_batch = next(iter(data_loader))

        print(f"--- データ拡張のデバッグ ---")
        print(f"バッチサイズ (IMS_PER_BATCH): {len(first_batch)}")

        # バッチ内の各画像を表示
        for i, data_dict in enumerate(first_batch):
            image = data_dict["image"].cpu().numpy() # TensorをNumPy配列に変換
            image = np.transpose(image, (1, 2, 0))   # C, H, W -> H, W, C (Detectron2の形式)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGRをRGBに変換 (Visualizer向け)

            print(f"画像 {i+1} のアノテーション数: {len(data_dict['instances'])}")
            # Visualizerを初期化
            visualizer = Visualizer(
                image, 
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                scale=0.8,
                font_size_scale=0.0,
                instance_mode=ColorMode.SEGMENTATION # マスク描画を強調
            )

            # draw_dataset_dictで正解マスクを描画
            vis = visualizer.draw_dataset_dict(data_dict)
            # 画像を表示
            window_name = f"Augmented Image {i+1} - GT Mask (Press any key to close)"
            cv2.imshow(window_name, vis.get_image()[:, :, ::-1])
            print(f"画像 {i+1} を表示しました。")

    # すべてのウィンドウが表示されるまで待機し、キーが押されたら終了
    print("表示ウィンドウを閉じるには、任意のキーを押してください...")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("--- デバッグ終了 ---")

    def build_train_loader(self, cfg):
        # デバッグフラグ (学習開始前に一度だけ実行)
        if hasattr(self, '_debug_run') and self._debug_run == False:
            self.debug_data_augmentation(cfg)
            self._debug_run = True # 実行済みフラグをセット
        # ここで custom_mapper を指定して学習用データローダーを作成する
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    # __init__を上書きしてデバッグフラグを設定
    def __init__(self, cfg):
        super().__init__(cfg)
        # デバッグコードを一度だけ実行するためのフラグ
        self._debug_run = False

if __name__ == "__main__":
    trainer = CustomTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True) # 必要に応じて以前のチェックポイントから再開
    trainer.train() # トレーニングを開始

    # print("データ拡張の確認を実行します。学習は開始されません。") # 拡張結果だけを見たい場合に実行
    # CustomTrainer.debug_data_augmentation(cfg)  # コメントを外す