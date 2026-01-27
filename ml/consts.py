from pydantic import BaseModel, Field
from typing import Literal

from pathlib import Path


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

BASE_DIR = Path(__file__).resolve().parent.parent
PATH_MODEL = BASE_DIR / "model" / "random_forest_model.joblib"
PATH_ENCODER = BASE_DIR / "model" / "encoder.joblib"
PATH_LB = BASE_DIR / "model" / "lb.joblib"


# use aliases for the hyphen python issue
class CensusInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")


class Config:
    populate_by_name = True
