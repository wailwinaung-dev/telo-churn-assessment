import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# 1. Load & Clean
print("Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df = df.drop(columns=['customerID'])

# 2. Encode Strings to Numbers (and save the encoders for the API)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

joblib.dump(label_encoders, 'encoders.pkl')

# 3. Split Data
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (Handling Imbalance)
print("Training XGBoost...")
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=3  
)
model.fit(X_train, y_train)

# 5. Evaluate (Using F1-Score as required)
predictions = model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, predictions):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test, predictions):.2f}")

# 6. Save Model
joblib.dump(model, 'churn_model.pkl')
print("Model saved as churn_model.pkl")