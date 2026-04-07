from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Admission.csv"
LOG_PATH = BASE_DIR / "logs" / "app.log"
MODEL_PATH = BASE_DIR / "model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

TARGET_COLUMN = "Admit_Chance"
THRESHOLD = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 123