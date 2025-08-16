import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')

# --- 1. Load Data ---

train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
df_train = pd.read_csv(train_path, index_col='datetime', parse_dates=True)

# Create a directory for EDA plots
output_dir = 'eda_plots'
os.makedirs(output_dir, exist_ok=True)

print("--- Exploratory Data Analysis (EDA) on Training Set ---")

# --- 2. Analysis & Plotting ---

# Basic Info
print("\n1. Basic Information & Statistics:")
print(df_train.describe())

# Identify column types
numerical_cols = df_train.select_dtypes(include=np.number).columns
categorical_cols = df_train.select_dtypes(include=['object']).columns

# Correlation Matrix and Heatmap
print("\n2. Generating Correlation Heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df_train[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()
print(f"  Saved to {output_dir}/correlation_heatmap.png")

# Histograms of Numerical Features
print("\n3. Generating Histograms for Numerical Features...")
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train[col], kde=True)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hist_{col}.png'))
    plt.close()
print(f"  Saved histograms to {output_dir}/")

# Time-Series Decomposition
print("\n4. Generating Time-Series Decomposition Plot...")
# Resample to daily for a clearer plot, handling any gaps
df_daily = df_train['PM2.5'].resample('D').mean().interpolate(method='linear')
decomposition = seasonal_decompose(df_daily, model='additive', period=365)

plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Seasonal Decomposition of PM2.5 (Daily)', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'seasonal_decomposition.png'))
plt.close()
print(f"  Saved to {output_dir}/seasonal_decomposition.png")

print("\n--- EDA Complete. Plots saved to 'eda_plots' directory. ---")
