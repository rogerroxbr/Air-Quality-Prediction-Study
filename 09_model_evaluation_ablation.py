import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

output_dir = 'evaluation_plots'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Load Data and Champion Model ---

test_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\test.csv'
df_test_raw = pd.read_csv(test_path, index_col='datetime', parse_dates=True)
df_test = df_test_raw.groupby(df_test_raw.index).mean(numeric_only=True).asfreq('H')

with open('best_model.pkl', 'rb') as f:
    champion_details = pickle.load(f)

champion_model = champion_details['model']
champion_features = champion_model.feature_name_

print("--- Final Model Evaluation and Ablation Study ---")
print(f"Loaded Champion Model: {type(champion_model).__name__}")

X_test = df_test[champion_features]
y_test = df_test['PM2.5']

X_test.fillna(method='ffill', inplace=True); X_test.fillna(method='bfill', inplace=True)
y_test.fillna(method='ffill', inplace=True); y_test.fillna(method='bfill', inplace=True)

# --- 2. Final Evaluation on Hold-Out Test Set ---

print("\n--- Evaluating Champion Model on Test Set ---")
predictions = champion_model.predict(X_test)

mse_final = mean_squared_error(y_test, predictions)
rmse_final = np.sqrt(mse_final)
mae_final = mean_absolute_error(y_test, predictions)
r2_final = r2_score(y_test, predictions)

print(f"  Final Test MSE: {mse_final:.4f}")
print(f"  Final Test RMSE: {rmse_final:.4f}")
print(f"  Final Test MAE: {mae_final:.4f}")
print(f"  Final Test R-squared: {r2_final:.4f}")

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)

print("\nGenerating Predictions vs. Actuals plot...")
plt.figure(figsize=(15, 7))
results_df.sample(n=500, random_state=42).sort_index().plot(ax=plt.gca(), style=['-', '--'])
plt.title('Predictions vs. Actual Values on Test Set (Sample)', fontsize=16)
plt.ylabel('PM2.5')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_vs_actuals.png'))
plt.close()
print(f"  Saved to {output_dir}/predictions_vs_actuals.png")

# --- 3. Ablation Study ---

print("\n--- Conducting Ablation Study by Feature Groups ---")

train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
df_train_raw = pd.read_csv(train_path, index_col='datetime', parse_dates=True)
df_train = df_train_raw.groupby(df_train_raw.index).mean(numeric_only=True).asfreq('H')

X_train = df_train[champion_features]
y_train = df_train['PM2.5']

X_train.fillna(method='ffill', inplace=True); X_train.fillna(method='bfill', inplace=True)
y_train.fillna(method='ffill', inplace=True); y_train.fillna(method='bfill', inplace=True)

feature_groups = {
    'pollutants': [f for f in ['PM10', 'SO2', 'NO2', 'CO', 'O3'] if f in champion_features],
    'meteorological': [f for f in ['TEMP', 'PRES', 'DEWP', 'WSPM'] if f in champion_features]
}

ablation_results = {}

for group_name, group_features in feature_groups.items():
    print(f"  Ablating group: '{group_name}'...")
    features_to_keep = [f for f in champion_features if f not in group_features]
    if not features_to_keep: continue

    ablated_model = type(champion_model)(**champion_model.get_params())
    ablated_model.fit(X_train[features_to_keep], y_train)
    predictions_ablated = ablated_model.predict(X_test[features_to_keep])
    mse_ablated = mean_squared_error(y_test, predictions_ablated)
    ablation_results[group_name] = mse_ablated - mse_final

print("\nGenerating Ablation Study plot...")
ablation_df = pd.DataFrame.from_dict(ablation_results, orient='index', columns=['MSE_Increase'])
ablation_df.sort_values('MSE_Increase', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=ablation_df.index, y=ablation_df.MSE_Increase)
plt.title('Ablation Study: Performance Drop by Feature Group', fontsize=16)
plt.ylabel('Increase in MSE')
plt.xlabel('Feature Group Removed')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_study_results.png'))
plt.close()
print(f"  Saved to {output_dir}/ablation_study_results.png")

print("\n--- Model Evaluation and Ablation Complete ---")