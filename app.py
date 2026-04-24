from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load('best_churn_model.pkl')

class Customer(BaseModel):
    gender: str; SeniorCitizen: int; Partner: str; Dependents: str
    tenure: int; PhoneService: str; MultipleLines: str; InternetService: str
    OnlineSecurity: str; OnlineBackup: str; DeviceProtection: str; TechSupport: str
    StreamingTV: str; StreamingMovies: str; Contract: str; PaperlessBilling: str
    PaymentMethod: str; MonthlyCharges: float; TotalCharges: float

@app.post("/predict")
def predict(data: Customer):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return {"Churn": "Yes" if prediction == 1 else "No", "Olasılık": f"%{prob*100:.2f}"}