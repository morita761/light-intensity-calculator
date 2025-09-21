# from PIL import Image
# import os

# image_dir = "../data/2img_val/" # 問題の評価用画像ディレクトリ
# print(f"Checking images in: {image_dir}")
# for filename in os.listdir(image_dir):
#     # 画像ファイルのみを対象にする
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
#         filepath = os.path.join(image_dir, filename)
#         try:
#             with Image.open(filepath) as img:
#                 img.verify() # 画像が完全に読み込めるか検証
#                 # print(f"  {filename}: OK") # 正常な場合は表示しない
#             img.close() # ファイルハンドルを閉じる
#         except Exception as e:
#             print(f"  ERROR: Problem with {filename} - {e}")


from PIL import Image
import os

image_dir = "../data/2img_val/"
problematic_files = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.tif', '.tiff')):
        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                img.verify() # Verify file integrity
                img.load()   # Load image data
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            problematic_files.append(filename)

if problematic_files:
    print("\n--- Problematic TIFF files found ---")
    for f in problematic_files:
        print(f)
else:
    print("\nAll TIFF files in the directory seem to be readable.")