import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

combined_df = pd.read_csv(r"C:\Users\jhanv\OneDrive\Documents\Assignments\ST CySec\Project\cleaned_combined.csv")

# Clean column names
combined_df.columns = combined_df.columns.str.strip()

# Now safely extract features and labels
X = combined_df.drop(['Label'], axis=1)
y = combined_df['Label']

# Step 2: Train-test split (stratify keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Quick variance and unique count check
for col in combined_df.columns:
    print(f"{col}: Unique = {combined_df[col].nunique()}, Variance = {combined_df[col].var()}")

overlap = pd.merge(X_train, X_test, how='inner')
print(f"Overlapping records: {len(overlap)}")


# Step 3: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Choose and train a classifier
# You can switch between RandomForestClassifier(), SVC(), or KNeighborsClassifier()
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model = SVC(kernel='rbf', C=1.0, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
