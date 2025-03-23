import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
def train_power_predictor_model(dataset_file):
    df = pd.read_csv(dataset_file)

    # Assume last column is target
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define K-Fold Cross-Validation
    k = 5  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store results
    xgb_mse_list = []
    tabnet_mse_list = []
    ensemble_mse_list = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)
        y_pred_xgb = xgb_model.predict(dval)

        # Train Pretrained TabNet Model
        tabnet_model = TabNetRegressor(verbose=0)
        tabnet_model.fit(X_train, y_train.reshape(-1, 1), max_epochs=50, patience=10, batch_size=128, virtual_batch_size=32)

        y_pred_tabnet = tabnet_model.predict(X_val).flatten()

        # Weighted Ensemble Prediction
        alpha = 0.7  # XGBoost weight
        beta = 0.3   # TabNet weight
        y_pred_final = (alpha * y_pred_xgb) + (beta * y_pred_tabnet)

        # Evaluate Performance
        mse_xgb = mean_squared_error(y_val, y_pred_xgb)
        mse_tabnet = mean_squared_error(y_val, y_pred_tabnet)
        mse_final = mean_squared_error(y_val, y_pred_final)

        xgb_mse_list.append(mse_xgb)
        tabnet_mse_list.append(mse_tabnet)
        ensemble_mse_list.append(mse_final)

    # Print Cross-Validation Results
    print(f"🔹 XGBoost Avg MSE: {np.mean(xgb_mse_list):.4f} ± {np.std(xgb_mse_list):.4f}")
    print(f"🔹 TabNet Avg MSE: {np.mean(tabnet_mse_list):.4f} ± {np.std(tabnet_mse_list):.4f}")
    print(f"🔹 Ensemble Model Avg MSE: {np.mean(ensemble_mse_list):.4f} ± {np.std(ensemble_mse_list):.4f}")

def __main__():
    parser = argparse.ArgumentParser(description="Process user information.")
    parser.add_argument("dataset", type=str, help="power predictor data set")

    args = parser.parse_args()

    dataset = args.dataset
    train_power_predictor_model(dataset)
    print("Model training completed :)")