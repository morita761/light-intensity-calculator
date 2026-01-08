import segmentation_models_pytorch as smp

# U-Net の初期化（ResNetベース、出力チャンネル1）
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid' # 後でBCEWithLogitsLossに入れる
)
