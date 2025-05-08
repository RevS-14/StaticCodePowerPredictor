import os

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load New Dataset
new_file_path = "output/power_dataset_verify.csv"  # Update with your new dataset path
new_df = pd.read_csv(new_file_path)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Data Preprocessing (Same steps as training)
new_df["power_mW"] = new_df["power_mW"].round().astype(int)
for col in new_df.select_dtypes(include=['object']).columns:
    try:
        new_df[col] = pd.to_numeric(new_df[col], errors='ignore')
    except:
        pass

# Feature Scaling (Use the same scaler from training)
from sklearn.preprocessing import MinMaxScaler
# scaler = joblib.load("resources/scaler.pkl")  # Load the same scaler used for training
# numeric_cols = new_df.select_dtypes(include=['number']).columns
# new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

# Extract Features & Target
X_new = new_df.iloc[10:20, :-1].values  # Select only first 100 rows for features
y_new = new_df.iloc[10:20, -1].values   # Select only first 100 rows for actual target values


print(f"saipavan {X_new.shape[1]}")
# Load Trained Models
xgb_model = xgb.Booster()
xgb_model.load_model("trained_models/xgb_model.json")
tabnet_model = TabNetRegressor()
tabnet_model.load_model("trained_models/tabnet_model.zip")
tabnet_model.device = "cpu"  # Force CPU execution
torch.cuda.empty_cache()

# Make Predictions
dnew = xgb.DMatrix(X_new)
y_pred_xgb = xgb_model.predict(dnew)
y_pred_tabnet = tabnet_model.predict(X_new).flatten()

# Weighted Ensemble Prediction
alpha = 0.7  # XGBoost weight
beta = 0.3   # TabNet weight
y_pred_final = (alpha * y_pred_xgb) + (beta * y_pred_tabnet)

print(f"predicted values: {y_pred_final}, original values: {y_new}")

# --------- Model Evaluation Metrics ---------
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error

    print(f"\n📊 {model_name} Performance on New Data:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

evaluate_model(y_new, y_pred_xgb, "XGBoost")
evaluate_model(y_new, y_pred_tabnet, "TabNet")
evaluate_model(y_new, y_pred_final, "Ensemble Model")

# --------- Visualization ---------

plt.figure(figsize=(10, 6))
plt.scatter(y_new, y_pred_final, alpha=0.5, color='blue', label="Predicted vs Actual")
max_val = max(max(y_new), max(y_pred_final))
plt.plot([0, max_val], [0, max_val], linestyle='--', color='red', label="Ideal Prediction Line")
plt.xlabel("Actual Power (mW)")
plt.ylabel("Predicted Power (mW)")
plt.title("Actual vs Predicted Power Consumption (New Data)")
plt.legend()
plt.grid(True)
plt.show()

# Error Histogram
errors = y_new - y_pred_final
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, color='purple', alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution (New Data)")
plt.grid(True)
plt.show()

# Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_final, errors, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Power (mW)")
plt.ylabel("Residual Error")
plt.title("Residual Plot (New Data)")
plt.grid(True)
plt.show()