from pathlib import Path

# stats_figures/ フォルダ（どこから実行しても同じ場所を指す）
ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / 'config'
FIGURE_SETTINGS_FILE = CONFIG_DIR / 'figure_settings.txt'
DATASETS_FILE = CONFIG_DIR / 'datasets.txt'

DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'              # 輝度測定結果のテキスト（入力）
PROCESSED_DIR = DATA_DIR / 'processed'  # ノーマライズ済みCSV（中間データ）

OUTPUT_DIR = ROOT / 'output'            # 図の出力先（スクリプト名ごとにサブフォルダ）
