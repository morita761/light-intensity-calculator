from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator # 追加
import os

# データセット登録

# register_coco_instances("horseshoe_dataset", {}, "train_annotations.json", "simpledataset/images/")
register_coco_instances("horseshoe_dataset_train", {}, "train_annotations_horseshoe_RGB.json", "../data/2img/")
register_coco_instances("horseshoe_dataset_val", {}, "val_annotations_horseshoe_RGB.json", "../data/2img_val/") # 追加

cfg = get_cfg()
# Mask R-CNN R50-FPN
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f1027c.pkl"
cfg.MODEL.WEIGHTS = "../data/model_final_f10217.pkl" # ダウンロードしたファイルの相対パス

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
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # horseshoe
cfg.TEST.EVAL_PERIOD = 50 # 50イテレーションごとに評価を実行（任意、デフォルトは0）

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

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
    
if __name__ == "__main__":
    trainer = CustomTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False) # 必要に応じて以前のチェックポイントから再開
    trainer.train() # トレーニングを開始
