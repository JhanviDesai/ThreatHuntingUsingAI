import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split

combined_df = pd.read_csv(r"C:\Users\jhanv\OneDrive\Documents\Assignments\ST CySec\Project\CICIDS2017_Cleaned_Combined.csv")

print("Available columns:", combined_df.columns.tolist())
combined_df.columns = combined_df.columns.str.strip()

# Step 1: Normalize labels
combined_df['Label'] = combined_df['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' or x == 'Benign' else 'Attack')

# Step 2: Drop the 'source_file' column
if 'source_file' in combined_df.columns:
    combined_df = combined_df.drop(columns=['source_file'])

# Step 3: Drop constant columns
nunique = combined_df.nunique()
constant_columns = nunique[nunique <= 1].index.tolist()
# Print how many constant columns were found
print(f"Number of constant columns: {len(constant_columns)}")

# Print the names of the constant columns
print("Constant columns:")
for col in constant_columns:
    print(f" - {col}")
combined_df = combined_df.drop(columns=constant_columns)
print(f"Removed constant columns: {constant_columns}")

# Step 4: Split features and target
X = combined_df.drop('Label', axis=1)
y = combined_df['Label']

# Encode target
y = LabelEncoder().fit_transform(y)  # 0 for Attack, 1 for Benign

# Step 5: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)  # Threshold can be tuned
X_var = selector.fit_transform(X)
X = pd.DataFrame(X_var, columns=X.columns[selector.get_support()])

# Step 6: Correlation Heatmap
#plt.figure(figsize=(12, 10))
#correlation_matrix = X.corr()
#sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
#plt.title("Feature Correlation Heatmap")
#plt.tight_layout()
#plt.show()

# Step 7: Feature Importance with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 20 features by Random Forest importance:")
print(importances.head(20))

# Step 8: ANOVA F-test
anova_selector = SelectKBest(score_func=f_classif, k=20)
anova_selector.fit(X, y)
anova_features = X.columns[anova_selector.get_support()]
print("\nTop 20 features by ANOVA F-test:")
print(anova_features)

# Optional: Plot top features from Random Forest
importances.head(20).plot(kind='barh', title='Top 20 Features (Random Forest Importance)')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()
