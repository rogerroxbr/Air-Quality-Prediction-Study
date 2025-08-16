import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import warnings
import time

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Other
import pickle

warnings.filterwarnings('ignore')

# --- 1. Load Data and Define Feature Set ---

train_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
df_train_raw = pd.read_csv(train_path, index_col='datetime', parse_dates=True)

print("Aggregating data from multiple stations by taking the mean...")
df_train = df_train_raw.groupby(df_train_raw.index).mean(numeric_only=True)
df_train = df_train.asfreq('H')

features = ['PM10', 'SO2', 'CO', 'O3', 'TEMP', 'DEWP']
target = 'PM2.5'

features = [f for f in features if f in df_train.columns]
X = df_train[features]
y = df_train[target]

X.fillna(method='ffill', inplace=True); X.fillna(method='bfill', inplace=True)
y.fillna(method='ffill', inplace=True); y.fillna(method='bfill', inplace=True)

tscv = TimeSeriesSplit(n_splits=5) # Using 5 splits for more robust stats

print("--- Model Training and Optimization (with Cost & Advanced Metrics) ---")
print(f"Using {len(features)} features: {features}")

# --- 2. ML Model Competition ---

# Define multiple scoring metrics
scoring = {
    'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
    'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'r2': make_scorer(r2_score)
}

models_to_run = {
    'LightGBM': (lgb.LGBMRegressor(random_state=42), {'n_estimators': Integer(100, 500), 'learning_rate': Real(0.01, 0.2, 'log-uniform'), 'num_leaves': Integer(20, 60)}),
    'XGBoost': (xgb.XGBRegressor(random_state=42), {'n_estimators': Integer(100, 500), 'learning_rate': Real(0.01, 0.2, 'log-uniform'), 'max_depth': Integer(3, 10)}),
    'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': Integer(100, 500), 'max_depth': Integer(5, 20), 'min_samples_leaf': Integer(2, 10)}),
    'CatBoost': (cb.CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False), {'iterations': Integer(100, 500), 'learning_rate': Real(0.01, 0.2, 'log-uniform'), 'depth': Integer(4, 10)}),
}

all_results = {}

def run_bayesian_optimization(estimator, search_spaces, model_name):
    print(f"\nOptimizing {model_name}...")
    
    opt = BayesSearchCV(estimator, search_spaces, n_iter=25, cv=tscv, scoring=scoring, refit='neg_mse', n_jobs=-1, random_state=42)
    
    start_time = time.time()
    opt.fit(X, y)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    # Store results
    results = {
        'model': opt.best_estimator_,
        'best_params': opt.best_params_,
        'training_time_seconds': training_time,
        'cv_results': opt.cv_results_ # Save detailed results for stats
    }
    
    # Extract mean scores
    mean_mse = -opt.cv_results_[f'mean_test_neg_mse'][opt.best_index_]
    mean_mae = -opt.cv_results_[f'mean_test_neg_mae'][opt.best_index_]
    mean_r2 = opt.cv_results_[f'mean_test_r2'][opt.best_index_]
    
    results['mse'] = mean_mse
    results['rmse'] = np.sqrt(mean_mse)
    results['mae'] = mean_mae
    results['r2'] = mean_r2

    all_results[model_name] = results
    print(f"  - Best {model_name} MSE: {mean_mse:.4f}")
    print(f"  - Training Time: {training_time:.2f} seconds")

for name, (estimator, space) in models_to_run.items():
    try:
        run_bayesian_optimization(estimator, space, name)
    except Exception as e:
        print(f"Could not run {name}. Error: {e}")

# --- 3. Final Summary ---
print("\n--- Optimization Summary ---")
summary_data = []
for name, res in all_results.items():
    summary_data.append({
        'Model': name,
        'MSE': res['mse'],
        'RMSE': res['rmse'],
        'MAE': res['mae'],
        'R2': res['r2'],
        'Time (s)': res['training_time_seconds']
    })

summary_df = pd.DataFrame(summary_data).set_index('Model').sort_values('MSE')
print(summary_df)

# --- 4. Save Champion Model and Full Results ---

# Save the detailed results dictionary for statistical analysis
with open('all_model_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("\nFull results for all models saved to all_model_results.pkl")

# Save the best model for the next script
best_model_name = summary_df.index[0]
best_model_details = all_results[best_model_name]
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model_details, f)
print(f"Champion model ({best_model_name}) saved to best_model.pkl")