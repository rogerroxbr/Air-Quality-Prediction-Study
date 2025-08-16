import pandas as pd

# Load the consolidated dataset
prsa_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\PRSA_consolidated.csv'
df_prsa = pd.read_csv(prsa_path)

# --- Initial Analysis ---
print("--- PRSA Dataset Info ---")
df_prsa.info()

print("\n--- Missing Values (PRSA) ---")
print(df_prsa.isnull().sum())

# --- Data Cleaning Steps ---

# 1. Drop the 'No' column as it's just an index
df_prsa = df_prsa.drop('No', axis=1)

# 2. Create a proper datetime column
df_prsa['datetime'] = pd.to_datetime(df_prsa[['year', 'month', 'day', 'hour']])

# 3. Set datetime as the index
df_prsa = df_prsa.set_index('datetime')

# 4. Drop the original year, month, day, hour columns
df_prsa = df_prsa.drop(['year', 'month', 'day', 'hour'], axis=1)

# 5. Handle Missing Values
# For numerical columns, we can use forward fill for time-series data
numeric_cols = df_prsa.select_dtypes(include='number').columns
df_prsa[numeric_cols] = df_prsa[numeric_cols].fillna(method='ffill')

# For the categorical 'wd' (wind direction), we can fill with the most frequent value
most_frequent_wd = df_prsa['wd'].mode()[0]
df_prsa['wd'] = df_prsa['wd'].fillna(most_frequent_wd)


print("\n--- Data after Cleaning ---")
df_prsa.info()

print("\n--- Missing Values after Cleaning (PRSA) ---")
print(df_prsa.isnull().sum())

# Save the cleaned data
cleaned_output_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\PRSA_cleaned.csv'
df_prsa.to_csv(cleaned_output_path)

print(f"\nCleaned PRSA data saved to {cleaned_output_path}")
