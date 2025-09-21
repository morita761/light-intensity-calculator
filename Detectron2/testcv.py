# import cv2
# # image = cv2.imread("../data/2img_val/up_stronger.tif", cv2.IMREAD_COLOR)
# image = cv2.imread("../data/2img_val/Fz-RFP_001_green.png", cv2.IMREAD_COLOR)
# if image is None:
#     print("Failed to read image.")
# else:
#     print("Image loaded successfully with shape:", image.shape)


from PIL import Image

try:
    # img = Image.open("../data/2img_val/up_stronger.tif")
    img = Image.open("../data/2img_val/Fz-RFP_001_green.png")
    img.load()  # 実際に画像データを読み込む
    print("TIFF 読み込み成功")
except Exception as e:
    print("読み込みエラー:", e)
