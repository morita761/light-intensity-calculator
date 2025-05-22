import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class HorseShoeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".tif", "_mask.tif"))

        # image = Image.open(img_path).convert("RGB")
        image = cv2.imread(mask_path)
        image = cv2.resize(image, self.size)
        # img_tensor = self.transform(image)
        img_tensor = torch.tensor(filled_mask / 255.0, dtype=torch.float32).unsqueeze(0)

        # マスク画像をOpenCVで読み取り
        mask_cv = cv2.imread(mask_path)
        mask_cv = cv2.resize(mask_cv, self.size)
        mask_hsv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2HSV)

        # 青色の範囲を抽出（HSV）
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([140, 255, 255])
        mask_bin = cv2.inRange(mask_hsv, lower_blue, upper_blue)

        # 輪郭抽出
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭の内側を白で塗りつぶすマスク作成
        filled_mask = np.zeros_like(mask_bin)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

        # マスクを float32 にして Tensor 化
        mask_tensor = torch.tensor(filled_mask / 255.0, dtype=torch.float32).unsqueeze(0)

        # # 読み込みはできていた
        # mask_np = mask_tensor.squeeze(0).numpy().astype(np.uint8) * 255  # 0 or 255 に変換
        # cv2.imshow("Mask (Blue Area)", mask_np)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print("mask extraction")
        return img_tensor, mask_tensor
