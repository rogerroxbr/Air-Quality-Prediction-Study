import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Load the cleaned dataset
cleaned_prsa_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\PRSA_cleaned.csv'
df = pd.read_csv(cleaned_prsa_path, index_col='datetime', parse_dates=True)

# Ensure the dataframe is sorted by datetime index
df.sort_index(inplace=True)

print("--- Time Series Cross-Validation Strategy ---")

# Define the TimeSeriesSplit strategy
# n_splits: number of splits (folds)
# test_size: number of samples in each test set (optional, if not specified, it's inferred)
# gap: number of samples to exclude from the end of each train set before the test set
tscv = TimeSeriesSplit(n_splits=5) # Example: 5 splits

print(f"Defined TimeSeriesSplit with n_splits={tscv.n_splits}")

# Demonstrate the splits
for i, (train_index, test_index) in enumerate(tscv.split(df)):
    print(f"\nFold {i+1}:")
    train_start_date = df.index[train_index[0]]
    train_end_date = df.index[train_index[-1]]
    test_start_date = df.index[test_index[0]]
    test_end_date = df.index[test_index[-1]]

    print(f"  Train set: {len(train_index)} samples, from {train_start_date} to {train_end_date}")
    print(f"  Test set: {len(test_index)} samples, from {test_start_date} to {test_end_date}")

print("\n--- Time Series Cross-Validation Strategy Defined and Demonstrated ---")
