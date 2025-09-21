from pathlib import Path
from PIL import Image
import json

input_dir = Path("../data/2img_val/")
output_dir = Path("../data/2img_val_png/")
output_dir.mkdir(exist_ok=True)

for tif_file in input_dir.glob("*.tif"):
    im = Image.open(tif_file)
    output_path = output_dir / (tif_file.stem + ".png")
    im.save(output_path)

# アノテーションの file_name も書き換える
with open("val_annotations_horseshoe_RGB.json", "r") as f:
    data = json.load(f)

for item in data["images"]:
    item["file_name"] = item["file_name"].replace(".tif", ".png")

with open("val_annotations_horseshoe_RGB_converted.json", "w") as f:
    json.dump(data, f)
