from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from utils.logger import get_logger
from typing import List
import pandas as pd
from ml.data import process_data
import matplotlib.pyplot as plt
import seaborn as sns


logger = get_logger(__name__)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("return trained model")
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # logger.info("returning predictions")
    return model.predict(X)


def evaluate_slices(
    model: ClassifierMixin,
    data: pd.DataFrame,
    cat_features: List[str],
    label: str,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
):
    """
    evaluate the performance on categorical features
    calculates also the overall metrics for a column
    and returns them as 2 dataframes

    :param model: trained ml model
    :type model: sklearn.base.ClassifierMixin
    :param data: dataset of testdata
    :type data: pandas dataframe
    :param cat_features: list[str]
    :type cat_features: list of categorical features to evaluate
    :param label: name of target column
    :type label: str
    :param encoder: fitted encoder for categorical features
    :type encoder: sklearn.preprocessing.OneHotEncoder
    :param lb: fitted label binarizer
    :type lb:  sklearn.preprocessing.LabelBinarizer
    """

    logger.info("Evaluating categorical features")
    slice_results = []

    for feature in cat_features:
        for value in data[feature].unique():
            slice_df = data[data[feature] == value]

            # Skip tiny slices or slices with single-class labels
            if len(slice_df) < 10 or len(slice_df[label].unique()) < 2:
                continue

            # Process slice
            X_slice, y_slice, _, _ = process_data(
                slice_df, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
            )

            # Make predictions
            preds = inference(model, X_slice)

            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            # Log slice-level metrics
            logger.info(f"{feature}={value} | n={len(slice_df)} | P={precision:.3f}, R={recall:.3f}, F1={fbeta:.3f}")

            # Save to list
            slice_results.append(
                {
                    "feature": feature,
                    "value": value,
                    "n_samples": len(slice_df),
                    "precision": precision,
                    "recall": recall,
                    "f1": fbeta,
                }
            )

        logger.info("------------------------------")

    # Convert to DataFrame
    slice_df = pd.DataFrame(slice_results)

    # Compute weighted averages per feature
    agg_df = (
        slice_df.groupby("feature")
        .apply(
            lambda g: pd.Series(
                {
                    "n_total": g["n_samples"].sum(),
                    "precision_avg": (g["precision"] * g["n_samples"]).sum() / g["n_samples"].sum(),
                    "recall_avg": (g["recall"] * g["n_samples"]).sum() / g["n_samples"].sum(),
                    "f1_avg": (g["f1"] * g["n_samples"]).sum() / g["n_samples"].sum(),
                }
            )
        )
        .reset_index()
    )

    logger.info("Column-level aggregated metrics:")
    for _, row in agg_df.iterrows():
        logger.info(
            f"{row['feature']} | n={row['n_total']} | "
            f"P={row['precision_avg']:.3f}, R={row['recall_avg']:.3f}, F1={row['f1_avg']:.3f}"
        )

    return slice_df, agg_df


def plot_slice_metrics_combined(slice_df: pd.DataFrame):
    """
    Plots Precision, Recall, and F1 for each category of each categorical feature.

    :param slice_df: DataFrame returned by evaluate_slices (slice-level)
    """
    sns.set_style("whitegrid")

    for feature in slice_df["feature"].unique():
        feature_df = slice_df[slice_df["feature"] == feature].sort_values("f1")

        # Melt the DataFrame to long format for seaborn
        plot_df = feature_df.melt(
            id_vars=["value"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score"
        )

        plt.figure(figsize=(12, 5))
        sns.barplot(x="value", y="score", hue="metric", data=plot_df, palette="Set2")
        plt.title(f"Performance per category for '{feature}'")
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.show()
