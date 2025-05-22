import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def predict_single_image(model, image_path, size=(256, 256)):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        pred = model(img_tensor)  # (1, 1, H, W)
        pred_mask = pred.squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # 表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.show()
