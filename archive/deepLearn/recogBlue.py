import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def load_image_and_mask(image_path, mask_path, size=(256, 256)):
    # 入力画像の読み込み
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])(img)

    # OpenCVでマスク画像読み込み
    mask_img_cv = cv2.imread(mask_path)
    mask_img_cv = cv2.resize(mask_img_cv, size)
    mask_img_hsv = cv2.cvtColor(mask_img_cv, cv2.COLOR_BGR2HSV)

    # 青色の範囲を定義（HSV）
    lower_blue = np.array([100, 100, 50])   # H, S, V
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(mask_img_hsv, lower_blue, upper_blue)  # 0 or 255

    # PyTorch tensorに変換（0.0 or 1.0）
    mask_tensor = torch.tensor(blue_mask / 255.0, dtype=torch.float32).unsqueeze(0)

    return img_tensor, mask_tensor


def load_image_and_mask_torch(image_path, mask_path, size=(256, 256)):
    # 入力画像の読み込み
    img = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("RGB")

    # リサイズ
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)

    # 青色 (0, 0, 255) を抽出して 1 にする
    mask_np = np.array(mask_img.resize(size))
    blue_mask = np.all(mask_np == [0, 0, 255], axis=-1).astype(np.float32)
    mask_tensor = torch.tensor(blue_mask).unsqueeze(0)  # (1, H, W)

    print(type(img_tensor))
    print(type(mask_tensor))

    return img_tensor, mask_tensor
