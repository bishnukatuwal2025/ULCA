import pandas as pd
from src.preprocess import prepare_target, drop_unnecessary_columns

def test_prepare_target():
    df = pd.DataFrame({"Admit_Chance": [0.92, 0.76, 0.80]})
    result = prepare_target(df)
    assert result["Admit_Chance"].tolist() == [1, 0, 1]

def test_drop_unnecessary_columns():
    df = pd.DataFrame({
        "Serial_No": [1, 2],
        "GRE_Score": [320, 315]
    })
    result = drop_unnecessary_columns(df)
    assert "Serial_No" not in result.columns