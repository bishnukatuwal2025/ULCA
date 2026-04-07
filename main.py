import joblib
from src.config import DATA_PATH, MODEL_PATH, SCALER_PATH
from src.data_loader import load_data
from src.preprocess import preprocess_pipeline
from src.model import build_model, train_model
from src.evaluate import evaluate_model

def main():
    df = load_data(DATA_PATH)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = preprocess_pipeline(df)

    model = build_model(
        hidden_layer_sizes=(3,),
        batch_size=50,
        max_iter=200,
        random_state=123,
        activation="tanh"
    )

    model = train_model(model, X_train_scaled, y_train)
    results = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    print("\nModel Evaluation")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("\nModel and scaler saved successfully.")

if __name__ == "__main__":
    main()