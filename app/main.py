from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import parse_obj_as

from io import StringIO
import pandas as pd
from ml.data import process_data
from ml.model import inference
from ml.consts import CATEGORICAL_FEATURES, PATH_MODEL, PATH_ENCODER, PATH_LB, LABEL_COLUMN, CensusInput
import joblib
from utils import utils

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
    df = pd.DataFrame([input_data.model_dump(by_alias=False)])

    # Load pretrained artifacts
    model = joblib.load(PATH_MODEL)
    encoder = joblib.load(PATH_ENCODER)
    lb = joblib.load(PATH_LB)

    # Process data
    X, _, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    prediction = inference(model, X)

    return {"prediction": prediction[0]}
