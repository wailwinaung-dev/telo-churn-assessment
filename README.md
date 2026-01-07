# Telco Customer Churn Prediction System

## Overview

This project predicts customer churn using the Telco Customer Churn dataset. It addresses the **class imbalance** problem (73% No vs 27% Yes) by utilizing **XGBoost** with a weighted scale (`scale_pos_weight=3`) and optimizes for **F1-Score** and **ROC-AUC** rather than simple accuracy.

The system is deployed as a REST API using **FastAPI**, enabling real-time predictions with strict data validation.

## Project Structure

- **`analysis.ipynb`**: Jupyter Notebook for Exploratory Data Analysis (EDA). Visualizes class imbalance and correlations (e.g., Fiber Optic users churn more).
- **`train.py`**: The training pipeline. It loads data, cleans it, handles class imbalance, trains the XGBoost model, and saves the artifacts (`churn_model.pkl` and `encoders.pkl`).
- **`app.py`**: The production-ready API. It serves the trained model using FastAPI and Pydantic for data validation.
- **`requirements.txt`**: List of Python dependencies.
- **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**: The raw dataset.

## Setup & Installation

1. **Prerequisites**
   Ensure you have Python installed (version 3.8+ recommended).

2. **Install Dependencies**
   Open your terminal in the project folder and run:
   ```bash
    pip install -r requirements.txt
   ```

## How to Start the Application

### Step 1: Run Data Analysis (Optional)

To view the data visualizations and understand the churn patterns:

1. Open `analysis.ipynb` in VS Code or Jupyter.
2. Click **"Run All"**.
3. The charts (Class Imbalance, Internet Service correlation) will appear inside the notebook.

### Step 2: Train the Model (Optional)

_Note: Pre-trained model files (`churn_model.pkl` and `encoders.pkl`) are already included in this repository. You can skip this step if you just want to run the API._

To retrain the model from scratch:

```bash
python train.py

```

_Output:_ You will see F1-Score metrics and the `.pkl` files will be updated.

### Step 3: Start the Prediction API

To start the web server, run:

```bash
uvicorn app:app --reload

```

You should see output indicating the server is running at `http://127.0.0.1:8000`.

### Step 4: Test the Prediction

The API comes with an interactive user interface (Swagger UI).

1. Open your browser and go to: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2. Click on the **`POST /predict`** bar.
3. Click **"Try it out"**.
4. **Copy and paste** one of the examples below into the "Request body" box.
5. Click **"Execute"**.

---

## Example JSON Payloads for Testing

### 1. High Risk Customer (Likely to Churn)

_Characteristics: Month-to-month contract, Fiber Optic, High monthly charges._

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.5,
  "TotalCharges": 191.0
}
```

### 2. Loyal Customer (Likely Safe)

_Characteristics: 2-Year Contract, 6+ years tenure, DSL._

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 70,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 65.0,
  "TotalCharges": 4550.0
}
```
