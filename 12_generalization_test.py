import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Champion Model Config and Italian Data ---

with open('best_model.pkl', 'rb') as f:
    champion_details = pickle.load(f)

champion_model_type = type(champion_details['model'])
champion_params = champion_details['model'].get_params()

ita_data_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\AirQualityUCI_cleaned.csv'
df_ita = pd.read_csv(ita_data_path, index_col='datetime', parse_dates=True)

print("--- Generalization Test: Training from Scratch on Italian Dataset ---")
print(f"Using champion model architecture: {champion_model_type.__name__}")

# --- 2. Prepare Italian Dataset ---

new_target = 'C6H6(GT)'
features = [col for col in df_ita.columns if col != new_target]

X = df_ita[features]
y = df_ita[new_target]

X.fillna(method='ffill', inplace=True); X.fillna(method='bfill', inplace=True)
y.fillna(method='ffill', inplace=True); y.fillna(method='bfill', inplace=True)

# --- 3. Temporal Train-Test Split ---

split_point = int(len(df_ita) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"Italian dataset split into {len(X_train)} training and {len(X_test)} test samples.")

# --- 4. Train and Evaluate Model ---

model_on_ita_data = champion_model_type(**champion_params)

print("\nTraining new model on Italian data...")
model_on_ita_data.fit(X_train, y_train)

print("Evaluating model on Italian test set...")
predictions = model_on_ita_data.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Generalization Test Results ---")
print(f"Target Variable: {new_target}")
print(f"Features Used: {features}")
print(f"Test Set MSE: {mse:.4f}")
print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set R-squared: {r2:.4f}")

print("\n--- Generalization Test Complete ---")