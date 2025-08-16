import pickle
import numpy as np
from scipy import stats
import pandas as pd

# --- 1. Load Detailed CV Results ---

print("--- Statistical Significance Analysis ---")

try:
    with open('all_model_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    print("Loaded detailed results from all_model_results.pkl")
except FileNotFoundError:
    print("Error: `all_model_results.pkl` not found.")
    print("Please run script 08_model_training_optimization.py first.")
    exit()

# --- 2. Identify Top Two Models ---

# Create a summary to find the best models based on mean MSE
summary_data = []
for name, res in all_results.items():
    summary_data.append({
        'Model': name,
        'MSE': res['mse']
    })
summary_df = pd.DataFrame(summary_data).set_index('Model').sort_values('MSE')

top_models = summary_df.head(2).index.tolist()

if len(top_models) < 2:
    print("Need at least two models to perform statistical comparison.")
    exit()

model_1_name = top_models[0]
model_2_name = top_models[1]

print(f"\nComparing top 2 models: 1st - {model_1_name} (MSE: {summary_df.loc[model_1_name]['MSE']:.4f}) vs. 2nd - {model_2_name} (MSE: {summary_df.loc[model_2_name]['MSE']:.4f})")

# --- 3. Extract Fold Scores ---

# The scores are stored as negative MSE, so we negate them
model_1_scores = -all_results[model_1_name]['cv_results']['mean_test_neg_mse']
model_2_scores = -all_results[model_2_name]['cv_results']['mean_test_neg_mse']

# --- 4. Perform Paired T-Test ---

# The paired t-test will tell us if the difference between the two models is significant
# Null Hypothesis (H0): The true mean difference between the models is zero.
# Alternative Hypothesis (H1): The true mean difference between the models is not zero.

t_statistic, p_value = stats.ttest_rel(model_1_scores, model_2_scores)

print("\n--- Paired T-Test Results ---")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# --- 5. Interpretation ---

alpha = 0.05 # Standard significance level

print("\n--- Conclusion ---")
if p_value < alpha:
    print(f"The p-value ({p_value:.4f}) is less than {alpha}.")
    print(f"We reject the null hypothesis. The difference in performance between {model_1_name} and {model_2_name} is statistically significant.")
else:
    print(f"The p-value ({p_value:.4f}) is greater than or equal to {alpha}.")
    print(f"We fail to reject the null hypothesis. There is not enough evidence to claim a significant performance difference between {model_1_name} and {model_2_name}.")

print("\n--- Statistical Analysis Complete ---")
