# import matplotlib.pyplot as plt
# from PIL import Image
# import torch
# from torchvision import transforms

# def visualize_prediction(model, image_path):
#     model.eval()
#     img = Image.open(image_path).convert("RGB").resize((256, 256))
#     tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred = model(tensor)
#         pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
#         binary_mask = (pred_mask > 0.5).astype(np.uint8)

#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Input")

#     plt.subplot(1, 2, 2)
#     plt.imshow(binary_mask, cmap="gray")
#     plt.title("Predicted Mask")
#     plt.show()
