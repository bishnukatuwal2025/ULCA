import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully from %s", file_path)
        return df
    except FileNotFoundError as exc:
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"Dataset not found at: {file_path}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading data")
        raise exc