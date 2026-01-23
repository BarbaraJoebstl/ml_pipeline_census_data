from unittest.mock import MagicMock
import numpy as np
from ml.model import inference, compute_model_metrics


def test_compute_model_metrics_basic():
    # prep mock data
    mock_y_true = np.array([1, 0, 1, 0, 1])
    mock_y_pred = np.array([1, 0, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(mock_y_true, mock_y_pred)

    # Check the types
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    # Check values manually (or via np.isclose for floats)
    # mock_y_true = [1,0,1,0,1], mock_y_pred = [1,0,0,0,1]
    # TP = 2 (positions 0,4), FP = 0, FN = 1 (position 2)
    # Precision = TP / (TP + FP) = 2 / 2 = 1.0
    # Recall = TP / (TP + FN) = 2 / 3 ≈ 0.6667
    # F1 = 2 * (precision*recall)/(precision+recall) ≈ 0.8

    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 2 / 3)
    assert np.isclose(fbeta, 0.8)


def test_inference_calls_predict():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1, 0, 1])

    # Fake input data
    mock_X = np.array([[0, 1], [1, 0], [0, 0]])

    # Call the inference function
    preds = inference(mock_model, mock_X)

    # Assertions
    mock_model.predict.assert_called_once_with(mock_X)
    assert isinstance(preds, np.ndarray)
    assert (preds == np.array([1, 0, 1])).all()
