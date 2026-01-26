from fastapi import FastAPI
import pandas as pd
from ml.data import process_data
from ml.model import inference
from ml.consts import CATEGORICAL_FEATURES, PATH_MODEL, PATH_ENCODER, PATH_LB
import joblib


app = FastAPI(title="Census Income Predictor API")


@app.get("/")
def root():
    return {"message": "Welcome to the Census Income Predictor API!"}


@app.post("/predict")
# todo create CensusInput datatype
def predict(input_data):
    """
    uses the pretrained model and returns the prediction

    :param input_data: _description_
    :type input_data: CensusInput
    :return: _description_
    :rtype: _type_
    """
    model = joblib.load(PATH_MODEL)
    encoder = joblib.load(PATH_ENCODER)
    lb = joblib.load(PATH_LB)

    data_dict = input_data.dict(by_alias=True)
    df = pd.DataFrame([data_dict])

    X, _, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)

    # Convert prediction back to original label
    pred_label = lb.inverse_transform(pred)[0]

    return {"prediction": int(pred_label)}
