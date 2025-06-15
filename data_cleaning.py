import pandas as pd
import glob
import numpy as np

# Load all CSV files
files = glob.glob(r'C:\Users\jhanv\OneDrive\Documents\Assignments\ST CySec\Project\archive\*.csv')
df_list = []

for file in files:
    df = pd.read_csv(file, low_memory=False)
    df['source_file'] = file.split('/')[-1]  # Optional: keep source file info
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
print("Combined shape:", combined_df.shape)

# Missing values
missing_values = combined_df.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

# Infinity values
has_inf = np.isinf(combined_df.select_dtypes(include=[np.number])).values.any()
print(f"Contains Infinity values: {'Yes' if has_inf else 'No'}")

# Replace inf/-inf with NaN
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

#Drop rows with any NaN (including from above step)
initial_shape = combined_df.shape
combined_df.dropna(inplace=True)
print(f"Dropped {initial_shape[0] - combined_df.shape[0]} rows with missing/infinite values.")

print(f"Total missing values: {combined_df.isnull().sum().sum()}")
print(f"Contains Infinity values: {'Yes' if np.isinf(combined_df.select_dtypes(include=[np.number])).values.any() else 'No'}")

print("\nFinal shape:", combined_df.shape)

# Optional: Save cleaned version
combined_df.to_csv('CICIDS2017_Cleaned_Combined.csv', index=False)