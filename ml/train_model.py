# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
from data import process_data
from ml.model import train_model, inference, compute_model_metrics, evaluate_slices
from ml.consts import CATEGORICAL_FEATURES, LABEL_COLUMN, PATH_ENCODER, PATH_MODEL, PATH_LB
from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("loading raw data")
# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CATEGORICAL_FEATURES, label=LABEL_COLUMN, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL_COLUMN,
    training=False,
    encoder=encoder,
    lb=lb,
)

logger.info("start to train model")
# Train and save a model.
model = train_model(X_train, y_train)

# inference on test set
preds = inference(model, X_test)
# calc metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {fbeta:.3f}")

joblib.dump(model, PATH_MODEL)
joblib.dump(encoder, PATH_ENCODER)
joblib.dump(lb, PATH_LB)

logger.info("Model and encoders saved to model/")

# run evalution on test data for slices
eval_slices, eval_columns = evaluate_slices(
    model,
    test,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    encoder,
    lb,
)

# not needed for prod
# plot_slice_metrics_combined(eval_slices)
