import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. Load Data and the Champion Model ---

# Load the champion model and its details
with open('best_model.pkl', 'rb') as f:
    champion_details = pickle.load(f)

champion_model = champion_details['model']
champion_features = champion_model.feature_name_

# Load train and test data
train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
test_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\test.csv'
df_train_raw = pd.read_csv(train_path, index_col='datetime', parse_dates=True)
df_test_raw = pd.read_csv(test_path, index_col='datetime', parse_dates=True)

# Aggregate data from multiple stations by taking the mean
df_train = df_train_raw.groupby(df_train_raw.index).mean(numeric_only=True).asfreq('H')
df_test = df_test_raw.groupby(df_test_raw.index).mean(numeric_only=True).asfreq('H')

# Prepare data using only the champion model's features
X_train = df_train[champion_features]
X_test = df_test[champion_features]

# Fill NaNs
X_train.fillna(method='ffill', inplace=True); X_train.fillna(method='bfill', inplace=True)
X_test.fillna(method='ffill', inplace=True); X_test.fillna(method='bfill', inplace=True)

# For SHAP, it's good practice to use a sample of the data for performance
X_train_sample = X_train.sample(n=100, random_state=42)
X_test_sample = X_test.sample(n=200, random_state=42)

print("--- Explainable AI (XAI) - SHAP Analysis ---")
print(f"Analyzing Champion Model: {type(champion_model).__name__}")

# --- 2. Initialize SHAP Explainer and Calculate Values ---

print("\nCalculating SHAP values...")
explainer = shap.Explainer(champion_model, X_train_sample)
shap_values = explainer(X_test_sample)

print("Calculating SHAP interaction values (can be slow)...")
shap_interaction_values = explainer.shap_interaction_values(X_test_sample)

# --- 3. Generate and Save SHAP Plots (as per PRD) ---

output_dir = 'shap_plots'
os.makedirs(output_dir, exist_ok=True)

print("\nGenerating SHAP Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP Summary Plot", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))
plt.close()
print(f"  Saved to {output_dir}/shap_summary_plot.png")

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_importance = pd.DataFrame(list(zip(X_test_sample.columns, mean_abs_shap)), columns=['feature', 'importance'])
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
top_features = feature_importance['feature'].head(4).tolist()

print(f"\nGenerating SHAP Dependence Plots for top features: {top_features}...")
for feature in top_features:
    plt.figure()
    shap.dependence_plot(feature, shap_values.values, X_test_sample, feature_names=X_test_sample.columns, show=False)
    plt.title(f"SHAP Dependence Plot for {feature}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'))
    plt.close()
print(f"  Saved to {output_dir}/")

print(f"\nGenerating SHAP Interaction Plots for top features: {top_features}...")
for feature in top_features:
    plt.figure()
    shap.summary_plot(shap_interaction_values, X_test_sample, feature_names=X_test_sample.columns, max_display=10, show=False)
    plt.title(f"SHAP Interaction Summary for {feature}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_interaction_{feature}.png'))
    plt.close()
print(f"  Saved to {output_dir}/")

print("\nGenerating SHAP Force Plot for a single instance...")
force_plot_html_path = os.path.join(output_dir, 'shap_force_plot.html')
force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test_sample.iloc[0,:], show=False)
shap.save_html(force_plot_html_path, force_plot)
print(f"  Saved to {force_plot_html_path}")

print("\n--- SHAP Analysis Complete ---")