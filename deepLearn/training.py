import dataclass as data
import unet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from sklearn.model_selection import train_test_split

def visualize_prediction(model, image_path, size=(1254, 624)):
    model.eval()
    img = Image.open(image_path).convert("RGB").resize(size)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # 表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.show()

image_list = sorted("dataset/images")
train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

# データ読み込み
height = 624
width = 1254
dataset = data.HorseShoeDataset("dataset/images", "dataset/masks", size=(1254, 624))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# モデル、損失関数、最適化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet.model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(type(device))
print(type(model))
print(type(criterion))
print(type(optimizer))

# テスト表示
img, mask = dataset[0]
plt.subplot(1, 2, 1)
# plt.imshow(img.permute(1, 2, 0))
plt.imshow(img.squeeze(0))
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze(0), cmap='gray')
plt.title("Filled Mask")
plt.show()

# sys.exit()


# 学習
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        # たとえば、モデル出力をマスクの形状にリサイズ（※非推奨だが手早く確認するには使える）
        outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

        print("Output shape:", outputs.shape)
        print("Mask shape:", masks.shape)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")


    # # 検証
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for imgs, masks in val_loader:
    #         ...
    #         val_loss += criterion(outputs, masks).item()

    # print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

visualize_prediction(model,"dataset/images/up_Fz_green_stronger.tif")