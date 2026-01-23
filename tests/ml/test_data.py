from unittest.mock import MagicMock
import numpy as np

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.data import process_data


def test_process_data_training():
    # Create a tiny fake DataFrame
    mock_df = pd.DataFrame(
        {"feature_cat": ["A", "B", "A", "B"], "feature_num": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]}
    )

    mock_cat_feat = ["feature_cat"]
    mock_label = "label"

    # Run process_data in training mode
    X_processed, y_processed, encoder, lb = process_data(
        mock_df, categorical_features=mock_cat_feat, label=mock_label, training=True
    )

    # Check types
    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y_processed, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

    # Check shapes
    # X_processed should have 2 continuous columns + 2 one-hot columns = 3 columns?
    # Continuous: 1 column, categorical one-hot: 2 unique values → total 1 + 2 = 3
    assert X_processed.shape == (4, 3)
    assert y_processed.shape == (4,)

    # Labels should be 0/1
    assert set(y_processed) == {0, 1}
