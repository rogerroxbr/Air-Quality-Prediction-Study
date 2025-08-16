import pandas as pd

# Load the cleaned dataset
cleaned_prsa_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\PRSA_cleaned.csv'
df = pd.read_csv(cleaned_prsa_path, index_col='datetime', parse_dates=True)

# --- Sort by datetime index before splitting ---
df.sort_index(inplace=True)

# --- Temporal Train-Test Split ---

# Determine the split point (80% for training, 20% for testing)
split_point = int(len(df) * 0.8)

# Split the data
df_train = df.iloc[:split_point]
df_test = df.iloc[split_point:]

# --- Verification ---
print("--- Data Split Verification ---")
print(f"Total records: {len(df)}")
print(f"Training records: {len(df_train)} ({len(df_train)/len(df):.2%})")
print(f"Test records: {len(df_test)} ({len(df_test)/len(df):.2%})")
print(f"Training data goes from {df_train.index.min()} to {df_train.index.max()}")
print(f"Test data goes from {df_test.index.min()} to {df_test.index.max()}")

# --- Save the datasets ---
train_output_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\train.csv'
test_output_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\test.csv'

df_train.to_csv(train_output_path)
df_test.to_csv(test_output_path)

print(f"\nTraining data saved to {train_output_path}")
print(f"Test data saved to {test_output_path}")