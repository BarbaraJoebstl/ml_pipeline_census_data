from fastapi import FastAPI

import pandas as pd
from ml.data import process_data
from ml.model import inference
from ml.consts import CATEGORICAL_FEATURES, PATH_MODEL, PATH_ENCODER, PATH_LB, CensusInput
import joblib


app = FastAPI(title="Census Income Predictor API")


@app.get("/")
def root():
    return {"message": "Welcome to the Census Income Predictor API!"}


@app.post("/predict")
# todo create CensusInput datatype
async def predict(input_data: CensusInput):
    """
    uses the pretrained model and returns the prediction
    has type validation to make sure only files with the correct columns are used.

    :param input_data: _description_
    :type input_data: CensusInput
    :return: _description_
    :rtype: _type_
    """

    # Convert validated Pydantic model to DataFrame
    df = pd.DataFrame([input_data.model_dump(by_alias=True)])

    # Load pretrained artifacts
    model = joblib.load(PATH_MODEL)
    encoder = joblib.load(PATH_ENCODER)
    lb = joblib.load(PATH_LB)

    # Process data
    X, _, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    prediction = inference(model, X)

    pred_label = lb.inverse_transform(prediction)  # converts 0/1 back to original class

    return f"It is predicted that the person earns {pred_label}."
