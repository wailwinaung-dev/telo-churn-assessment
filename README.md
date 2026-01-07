# Telco Customer Churn Prediction System

## Overview

This project predicts customer churn using the Telco Customer Churn dataset. It addresses the **class imbalance** problem by utilizing XGBoost with a weighted scale (`scale_pos_weight`) and optimizes for **F1-Score**.

## Project Structure

- `analysis.ipynb`: EDA (Exploratory Data Analysis) visualizing churn correlations.
- `train.py`: Training pipeline that cleans data, handles imbalance, and serializes the model.
- `app.py`: FastAPI implementation serving the model via REST API.

## Setup & Usage

1. **Install Requirements**
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib matplotlib seaborn
   ```
2. **Start the Prediction API**
   ```bash
   uvicorn app:app --reload
   ```
3. **Test the Prediction**
   The API comes with an interactive user interface (Swagger UI).

   Open your browser and go to: http://127.0.0.1:8000/docs
