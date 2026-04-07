from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    logger.info("Train Accuracy: %.4f", train_acc)
    logger.info("Test Accuracy: %.4f", test_acc)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report": report
    }