import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the training dataset
train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
df_train = pd.read_csv(train_path, index_col='datetime', parse_dates=True)

# --- Prepare Data ---
target = 'PM2.5'
features = df_train.select_dtypes(include=np.number).columns.drop(target)
X = df_train[features]
y = df_train[target]

# Handle potential NaN values that could cause issues
y.fillna(method='ffill', inplace=True)
y.fillna(method='bfill', inplace=True)
X.fillna(method='ffill', inplace=True)
X.fillna(method='bfill', inplace=True)

# Define the cross-validation strategy consistently
tscv = TimeSeriesSplit(n_splits=5)

print("--- Feature Selection: Wrapper & Embedded Methods ---")

# --- 1. Recursive Feature Elimination with Cross-Validation (RFECV) ---
print("\n1. RFECV to find the optimal number of features:")
estimator = lgb.LGBMRegressor(random_state=42)
selector_rfecv = RFECV(estimator=estimator, step=1, cv=tscv, scoring='neg_mean_squared_error', min_features_to_select=3)
selector_rfecv.fit(X, y)
selected_features_rfecv = X.columns[selector_rfecv.support_]
print(f"  Features selected by RFECV ({selector_rfecv.n_features_} features): {list(selected_features_rfecv)}")

# --- 2. Lasso with Cross-Validation (LassoCV) ---
print("\n2. LassoCV to find the optimal alpha and select features:")
# Scale data for Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use LassoCV to find the best alpha
lasso_cv = LassoCV(cv=tscv, random_state=42)
lasso_cv.fit(X_scaled, y)

# Get features with non-zero coefficients
selected_features_lasso = X.columns[lasso_cv.coef_ != 0]
print(f"  LassoCV found optimal alpha: {lasso_cv.alpha_:.4f}")
print(f"  Features selected by LassoCV ({len(selected_features_lasso)} features): {list(selected_features_lasso)}")

# --- Baseline Model Evaluation ---
print("\n--- Baseline LightGBM Evaluation with Selected Features ---")

def evaluate_model(X_eval, y_eval, model_name, features_list):
    print(f"\nEvaluating {model_name} with {len(features_list)} features: {features_list}")
    mse_scores = []
    # Use a fresh CV split for evaluation
    eval_tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(eval_tscv.split(X_eval)):
        X_train, X_test = X_eval.iloc[train_idx], X_eval.iloc[test_idx]
        y_train, y_test = y_eval.iloc[train_idx], y_eval.iloc[test_idx]

        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train, y_train)
        predictions = lgbm.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)
    avg_mse = np.mean(mse_scores)
    print(f"  Average MSE for {model_name}: {avg_mse:.4f}")
    return avg_mse

# Evaluate each feature set
results = {}
if len(selected_features_rfecv) > 0:
    results['RFECV'] = evaluate_model(X[selected_features_rfecv], y, "RFECV Features", selected_features_rfecv)

if len(selected_features_lasso) > 0:
    # IMPORTANT: Evaluate on original (unscaled) data
    results['LassoCV'] = evaluate_model(X[selected_features_lasso], y, "LassoCV Features", selected_features_lasso)

# --- Conclusion ---
if results:
    winner = min(results, key=results.get)
    print(f"\n--- Winner: {winner} ---")
    print(f"The best feature set is from the '{winner}' method with an average MSE of {results[winner]:.4f}.")

print("\n--- Feature Selection Wrapper/Embedded Methods Complete ---")