import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Create a directory for uncertainty plots
output_dir = 'uncertainty_plots'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Load Data and the Champion Model Config ---

# Load the champion model details
with open('best_model.pkl', 'rb') as f:
    champion_details = pickle.load(f)

champion_model = champion_details['model']
champion_params = champion_model.get_params()
champion_features = champion_model.feature_name_

# Remove params that are not compatible with quantile objective
params_to_remove = ['objective', 'loss', 'metric']
for p in params_to_remove:
    if p in champion_params: del champion_params[p]

# Load train and test data
train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
test_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\test.csv'
df_train_raw = pd.read_csv(train_path, index_col='datetime', parse_dates=True)
df_test_raw = pd.read_csv(test_path, index_col='datetime', parse_dates=True)

# Aggregate data from multiple stations by taking the mean
df_train = df_train_raw.groupby(df_train_raw.index).mean(numeric_only=True).asfreq('H')
df_test = df_test_raw.groupby(df_test_raw.index).mean(numeric_only=True).asfreq('H')

# Prepare data
X_train = df_train[champion_features]; y_train = df_train['PM2.5']
X_test = df_test[champion_features]; y_test = df_test['PM2.5']

# Fill NaNs
X_train.fillna(method='ffill', inplace=True); X_train.fillna(method='bfill', inplace=True)
y_train.fillna(method='ffill', inplace=True); y_train.fillna(method='bfill', inplace=True)
X_test.fillna(method='ffill', inplace=True); X_test.fillna(method='bfill', inplace=True)
y_test.fillna(method='ffill', inplace=True); y_test.fillna(method='bfill', inplace=True)

print("--- Uncertainty Quantification using Champion Model's Hyperparameters ---")

# --- 2. Train Quantile Regression Models ---
quantiles = [0.05, 0.5, 0.95]
quantile_models = {}

print("\nTraining models for different quantiles...")
for q in quantiles:
    model = type(champion_model)(objective='quantile', alpha=q, **champion_params)
    model.fit(X_train, y_train)
    quantile_models[q] = model

# --- 3. Make Predictions and Form Prediction Intervals ---
print("\nMaking predictions and forming prediction intervals...")
predictions_df = pd.DataFrame(index=y_test.index)
predictions_df['Actual'] = y_test

for q, model in quantile_models.items():
    predictions_df[f'q_{q}'] = model.predict(X_test)

predictions_df.rename(columns={'q_0.05': 'Lower Bound', 'q_0.5': 'Median', 'q_0.95': 'Upper Bound'}, inplace=True)

# --- 4. Evaluate and Plot Prediction Intervals ---
print("\nEvaluating and Plotting Prediction Intervals...")

coverage = ((predictions_df['Actual'] >= predictions_df['Lower Bound']) & (predictions_df['Actual'] <= predictions_df['Upper Bound'])).mean()
print(f"  Prediction Interval Coverage (Target: 90%): {coverage:.2%}")

# Plotting a sample of the prediction intervals
plot_sample = predictions_df.sample(n=200, random_state=42).sort_index()

plt.figure(figsize=(15, 7))
plt.plot(plot_sample.index, plot_sample['Actual'], label='Actual', color='black', marker='.', linestyle='None')
plt.plot(plot_sample.index, plot_sample['Median'], label='Median Prediction', color='red')
plt.fill_between(plot_sample.index, plot_sample['Lower Bound'], plot_sample['Upper Bound'], color='red', alpha=0.2, label='90% Prediction Interval')
plt.title('Uncertainty Quantification: Prediction Intervals (Sample)', fontsize=16)
plt.ylabel('PM2.5')
plt.xlabel('Datetime')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'uncertainty_intervals_plot.png'))
plt.close()
print(f"  Saved to {output_dir}/uncertainty_intervals_plot.png")

print("--- Uncertainty Quantification Complete ---")