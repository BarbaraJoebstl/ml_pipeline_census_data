# ml/consts.py

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL_COLUMN = "salary"

PATH_MODEL = "../model/random_forest_model.joblib"
PATH_ENCODER = "../model/encoder.joblib"
PATH_LB = "../model/lb.joblib"
