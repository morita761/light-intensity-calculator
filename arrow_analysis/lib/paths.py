from pathlib import Path

# arrow_analysis/ フォルダ（どこから実行しても同じ場所を指す）
ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / 'config'
FIGURE_SETTINGS_FILE = CONFIG_DIR / 'figure_settings.txt'
IMAGE_SETS_FILE = CONFIG_DIR / 'image_sets.txt'

OUTPUT_DIR = ROOT / 'output'  # 図の出力先（スクリプト名ごとにサブフォルダ）
