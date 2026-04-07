import pandas as pd

def make_single_prediction(model, scaler, input_data: dict) -> int:
    df = pd.DataFrame([input_data])

    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")

    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

    expected_columns = [
        "GRE_Score",
        "TOEFL_Score",
        "SOP",
        "LOR",
        "CGPA",
        "University_Rating_1",
        "University_Rating_2",
        "University_Rating_3",
        "University_Rating_4",
        "University_Rating_5",
        "Research_0",
        "Research_1",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    return int(prediction)