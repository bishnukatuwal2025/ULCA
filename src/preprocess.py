import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.config import TARGET_COLUMN, THRESHOLD, TEST_SIZE, RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} column not found in dataset.")

    df[TARGET_COLUMN] = (df[TARGET_COLUMN] >= THRESHOLD).astype(int)
    logger.info("Target column converted to binary using threshold %.2f", THRESHOLD)
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Serial_No" in df.columns:
        df = df.drop(columns=["Serial_No"])
        logger.info("Dropped Serial_No column")
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "University_Rating" in df.columns:
        df["University_Rating"] = df["University_Rating"].astype("object")
    if "Research" in df.columns:
        df["Research"] = df["Research"].astype("object")

    df = pd.get_dummies(
        df,
        columns=["University_Rating", "Research"],
        dtype=int
    )

    logger.info("Categorical encoding completed")
    return df

def split_features_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} not found after preprocessing.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info("Train-test split completed")
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling completed using MinMaxScaler")
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(df: pd.DataFrame):
    df = prepare_target(df)
    df = drop_unnecessary_columns(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler