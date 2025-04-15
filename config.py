from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[0]


SOUNDS_DATA_DIR_PATH = ROOT_DIR / "data" / "Guitar Dataset"
DATA_SPLIT_SAVE_DIR_PATH = ROOT_DIR / "data" / "prepared_data"
MODEL_DIR_PATH = ROOT_DIR / "model"
MODEL_HISTORY_DIR_PATH = ROOT_DIR / "data" / "model_history"
DATAFRAMES_DIR_PATH = ROOT_DIR / "data" / "dataframes"

print(DATA_SPLIT_SAVE_DIR_PATH)