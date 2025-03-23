import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.nn.functional import dropout

# Load data
data = pd.read_csv("../output/power_dataset_new.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split into train and test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize for XGBoost
scaler_xgb = StandardScaler()
X_train_full_scaled = scaler_xgb.fit_transform(X_train_full)
X_test_scaled = scaler_xgb.transform(X_test)

# MinMax Scale for TabNet
scaler_tabnet = MinMaxScaler()
X_train_full_tabnet = scaler_tabnet.fit_transform(X_train_full)
X_test_tabnet = scaler_tabnet.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
xgb_mse_list = []
tabnet_mse_list = []
ensemble_mse_list = []

# XGBoost Hyperparameter Tuning
xgb_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "n_estimators": [300, 500, 1000, 1500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    estimator=xgb.XGBRegressor(objective="reg:squarederror"),
    param_grid=xgb_params,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1
)
grid_xgb.fit(X_train_full_scaled, y_train_full)
best_xgb_params = grid_xgb.best_params_
print(f"Best XGBoost Params: {best_xgb_params}")

for k_fold_train_idx, k_fold_test_idx in kf.split(X_train_full_scaled):
    X_train, X_valid = X_train_full_scaled[k_fold_train_idx], X_train_full_scaled[k_fold_test_idx]
    y_train, y_valid = y_train_full[k_fold_train_idx], y_train_full[k_fold_test_idx]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)

    xgb_model = xgb.train(best_xgb_params, d_train, num_boost_round=300)
    y_predictions_xgb = xgb_model.predict(d_valid)

    # TabNet Hyperparameter Tuning
    tabnet_model = TabNetRegressor(
        n_d=32,  # Increase depth
        n_a=32,
        n_steps=5,# Increase attention size
        gamma=1.2,  # Helps generalization
        seed=42,
        momentum=0.2,
        lambda_sparse=0.001
    )
    tabnet_model.fit(
        X_train, y_train.reshape(-1, 1),
        max_epochs=1000, patience=20,
        batch_size=256, virtual_batch_size=100
    )
    y_predictions_tabnet = tabnet_model.predict(X_valid).flatten()

    # Weighted Ensemble
    alpha = 0.6  # More weight to XGBoost
    beta = 0.4
    y_predictions_final = (alpha * y_predictions_xgb) + (beta * y_predictions_tabnet)

    mse_xgb = mean_squared_error(y_valid, y_predictions_xgb)
    mse_tabnet = mean_squared_error(y_valid, y_predictions_tabnet)
    mse_final = mean_squared_error(y_valid, y_predictions_final)

    xgb_mse_list.append(mse_xgb)
    tabnet_mse_list.append(mse_tabnet)
    ensemble_mse_list.append(mse_final)

# Print final results
print(f"XGBoost Avg MSE: {np.mean(xgb_mse_list):.4f} ± {np.std(xgb_mse_list):.4f}")
print(f"TabNet Avg MSE: {np.mean(tabnet_mse_list):.4f} ± {np.std(tabnet_mse_list):.4f}")
print(f"Ensemble Model Avg MSE: {np.mean(ensemble_mse_list):.4f} ± {np.std(ensemble_mse_list):.4f}")