from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load Model & Encoders
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')

app = FastAPI()

# DTO Schema
class CustomerDTO(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerDTO):
    try:
        # Convert DTO to DataFrame
        df = pd.DataFrame([data.dict()])

        # Preprocess: Encode text inputs using saved encoders
        for col, le in encoders.items():
            if col in df.columns and col != 'Churn':
                # Safe transform (handles unknown values)
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

        # Predict
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df).max()

        return {
            "prediction": "Churn (Risk)" if prediction == 1 else "No Churn (Safe)",
            "confidence": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))