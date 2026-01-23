# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, inference, compute_model_metrics, evaluate_slices, plot_slice_metrics_combined
from logger import get_logger

logger = get_logger(__name__)

logger.info("loading raw data")
# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
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

joblib.dump(model, "../model/random_forest_model.joblib")
joblib.dump(encoder, "../model/encoder.joblib")
joblib.dump(lb, "../model/lb.joblib")

logger.info("Model and encoders saved to model/")

# run evalution on test data for slices
eval_slices, eval_columns = evaluate_slices(
    model,
    test,
    cat_features,
    "salary",
    encoder,
    lb,
)

# not needed for prod
# plot_slice_metrics_combined(eval_slices)
