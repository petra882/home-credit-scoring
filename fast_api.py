from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from catboost import CatBoostClassifier


encoder = joblib.load("model/encoder.pkl")
review_model = joblib.load("model/model.pkl")

class CreditApplication(BaseModel):
    age_years: float
    credit_to_income: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_CONTRACT_TYPE: str
    NAME_INCOME_TYPE: str
    CODE_GENDER: str


app = FastAPI(title="Credit Risk API")


@app.get("/")
async def root():
    return {"message": "Credit Risk Prediction API"}


@app.post("/predict")
async def predict(request: CreditApplication):
    numeric_features = [[
        request.age_years,
        request.credit_to_income,
        request.EXT_SOURCE_1,
        request.EXT_SOURCE_2,
        request.EXT_SOURCE_3
    ]]

    str_features = [[
        request.NAME_CONTRACT_TYPE,
        request.NAME_INCOME_TYPE,
        request.CODE_GENDER
    ]]

    str_features_encoded = encoder.transform(str_features)
    features = np.hstack([numeric_features, str_features_encoded])

    prediction = review_model.predict(features)
    return{
        "prediction": int(prediction[0])
    }
