from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import torch

# --- Detectron2 設定 ---
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

# --- 推論器 ---
predictor = DefaultPredictor(cfg)

# --- 画像読み込み ---
# image = cv2.imread("./simpledataset/images/01.png")
image = cv2.imread("../data/img/004.tif")
if image is None:
    raise FileNotFoundError("画像 simpledataset/images/003.png が読み込めませんでした")

# cv2.imshow('Sample Image',image)
# cv2.waitKey()

# image = image.astype("float32").transpose(2, 0, 1)
# image = torch.as_tensor(image, dtype=torch.float32)


# --- 推論 ---
outputs = predictor(image)

# --- メタデータ登録 ---
MetadataCatalog.get("horseshoe_dataset").set(thing_classes=["horseshoe"])

# --- 可視化と保存 ---
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("horseshoe_dataset"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output/result_horseshoe_004.png", out.get_image()[:, :, ::-1])
