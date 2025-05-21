import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
# image = cv2.imread('up_Fz_green_stronger.tif', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('up_Fz_green_stronger.tif')

# 環境に依存しないようにこれでgray scaleにする
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ヒストグラムの作成（引数の'image'を[ ]で囲うことを忘れないで下さい）
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# ヒストグラムの可視化
plt.rcParams["figure.figsize"] = [12,7.5]                         # 表示領域のアスペクト比を設定
plt.subplots_adjust(left=0.01, right=0.95, bottom=0.10, top=0.95) # 余白を設定
plt.subplot(221)                                                  # 1行2列の1番目の領域にプロットを設定
plt.imshow(image, cmap='gray')                                    # 画像をグレースケールで表示
plt.axis("off")                                                   # 軸目盛、軸ラベルを消す
plt.subplot(222)                                                  # 1行2列の2番目の領域にプロットを設定
plt.plot(histogram)                                               # ヒストグラムのグラフを表示
plt.xlabel('Brightness')                                          # x軸ラベル(明度)
plt.ylabel('Count')                                               # y軸ラベル(画素の数)

# CLAHEのパラメータ
clip_limit = 2.0
tile_grid_size = (8, 8)

# CLAHEオブジェクトを作成
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

# CLAHEを適用
result = clahe.apply(image)

# CLAHE適応後画像のヒストグラムの作成
histogram = cv2.calcHist([result], [0], None, [256], [0, 256])
# ヒストグラムの可視化
plt.subplot(223)                                                  # 1行2列の1番目の領域にプロットを設定
plt.imshow(result, cmap='gray')                                   # 画像をグレースケールで表示
plt.axis("off")                                                   # 軸目盛、軸ラベルを消す
plt.subplot(224)                                                  # 1行2列の2番目の領域にプロットを設定
plt.plot(histogram)                                               # ヒストグラムのグラフを表示
plt.xlabel('Brightness')                                          # x軸ラベル(明度)
plt.ylabel('Count')                                               # y軸ラベル(画素の数)
plt.show()