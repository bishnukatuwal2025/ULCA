from sklearn.neural_network import MLPClassifier
from src.logger import get_logger

logger = get_logger(__name__)

def build_model(
    hidden_layer_sizes=(3,),
    batch_size=50,
    max_iter=200,
    random_state=123,
    activation="tanh"
):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        activation=activation
    )
    logger.info(
        "MLP model created | hidden_layer_sizes=%s | batch_size=%s | max_iter=%s | activation=%s",
        hidden_layer_sizes, batch_size, max_iter, activation
    )
    return model

def train_model(model, X_train_scaled, y_train):
    model.fit(X_train_scaled, y_train)
    logger.info("Model training completed")
    return model