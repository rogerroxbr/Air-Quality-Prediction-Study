import pandas as pd
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training dataset
train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
df_train = pd.read_csv(train_path, index_col='datetime', parse_dates=True)

# --- Prepare Data ---
# Define target and automatically select numeric features
target = 'PM2.5'
features = df_train.select_dtypes(include=np.number).columns.drop(target)

X = df_train[features]
y = df_train[target]

# Handle potential NaN values in target that could cause issues
y.fillna(method='ffill', inplace=True)
y.fillna(method='bfill', inplace=True)

print("--- Feature Selection: Filter Methods ---")

# --- 1. Correlation Coefficient with Target --- 
print("\n1. Correlation Coefficient with Target (Pearson):")
correlations = X.corrwith(y).abs().sort_values(ascending=False)
threshold_corr = 0.2 # Using a slightly higher threshold
selected_features_corr = correlations[correlations > threshold_corr].index.tolist()
print(f"  Features selected by Correlation > {threshold_corr}: {selected_features_corr}")

# --- 2. Mutual Information ---
print("\n2. Mutual Information with Target:")
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, name="MI_Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# Select top N features based on MI score
N_FEATURES_MI = 5 # Example: select top 5 features
selected_features_mi = mi_scores.head(N_FEATURES_MI).index.tolist()
print(f"  Top {N_FEATURES_MI} features selected by Mutual Information: {selected_features_mi}")

# --- Baseline Model Evaluation ---
print("\n--- Baseline LightGBM Evaluation with Selected Features ---")

tscv = TimeSeriesSplit(n_splits=5)

def evaluate_model(X_selected, y_selected, model_name, features_list):
    print(f"\nEvaluating {model_name} with {len(features_list)} features: {features_list}")
    mse_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y_selected.iloc[train_idx], y_selected.iloc[test_idx]

        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train, y_train)
        predictions = lgbm.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)
        # print(f"  Fold {fold+1} MSE: {mse:.4f}")
    avg_mse = np.mean(mse_scores)
    print(f"  Average MSE for {model_name}: {avg_mse:.4f}")
    return avg_mse

# Evaluate each feature set
results = {}

if len(selected_features_corr) > 0:
    results['Correlation'] = evaluate_model(X[selected_features_corr], y, "Correlation Features", selected_features_corr)
else:
    print("  No features selected by Correlation Coefficient to evaluate.")

if len(selected_features_mi) > 0:
    results['Mutual_Info'] = evaluate_model(X[selected_features_mi], y, "Mutual Information Features", selected_features_mi)
else:
    print("  No features selected by Mutual Information to evaluate.")

# --- Conclusion ---
if results:
    winner = min(results, key=results.get)
    print(f"\n--- Winner: {winner} ---")
    print(f"The best feature set is from the '{winner}' method with an average MSE of {results[winner]:.4f}.")

print("\n--- Feature Selection Filter Methods Complete ---")