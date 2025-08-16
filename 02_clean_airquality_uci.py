import pandas as pd
import numpy as np

# --- 1. Load and Clean Original Dataset ---

airquality_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\air+quality\AirQualityUCI.csv'
df_aq = pd.read_csv(airquality_path, sep=';', decimal=',')

df_aq = df_aq.iloc[:, :-2]
df_aq.replace(-200, np.nan, inplace=True)
df_aq['datetime'] = pd.to_datetime(df_aq['Date'] + ' ' + df_aq['Time'], format='%d/%m/%Y %H.%M.%S', dayfirst=True)
df_aq = df_aq.set_index('datetime')
df_aq = df_aq.drop(['Date', 'Time'], axis=1)
df_aq = df_aq.interpolate(method='linear')
df_aq.bfill(inplace=True)

print("--- Original AirQualityUCI data cleaned ---")

# --- 2. Harmonize Columns with PRSA Dataset ---

# Adding C6H6(GT) as requested for the generalization test.
column_mapping = {
    'CO(GT)': 'CO',
    'NO2(GT)': 'NO2',
    'T': 'TEMP',
    'RH': 'RH',
    'C6H6(GT)': 'C6H6(GT)' # Keep the new target variable
}

df_aq.rename(columns=column_mapping, inplace=True)

# Select only the harmonized columns plus the new target
harmonized_columns = list(column_mapping.values())
df_harmonized = df_aq[harmonized_columns]

print("\n--- Harmonization Complete ---")
print("Columns selected and renamed for compatibility:")
print(df_harmonized.columns)

# --- 3. Save the Cleaned and Harmonized Data ---

cleaned_output_path = r'c:\Users\Roger\Documents\Projetos\Air quality dados\AirQualityUCI_cleaned.csv'
df_harmonized.to_csv(cleaned_output_path)

print(f"\nCleaned and Harmonized AirQualityUCI data saved to {cleaned_output_path}")
