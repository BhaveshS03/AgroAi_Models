# %% Install dependencies (if not already installed)
# !pip install xgboost scikit-learn matplotlib seaborn pandas

# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load and preprocess data
df = pd.read_csv("ops.csv")

# Drop rows where target is NaN or 'BARE'
df = df.dropna(subset=['VARIETY'])
df = df[df['VARIETY'] != 'BARE']

# Features and target
features = ['RF', 'ET', 'MAX', 'MIN', 'MTMP', 'RH1', 'RH2']
X = df[features]
y = df['VARIETY'].astype(str)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Number of samples after filtering:", len(y_encoded))
print("Number of classes:", len(le.classes_))

# %% Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# %% Build and train XGBoost model
model = XGBClassifier(
    n_estimators=300,         # number of trees
    learning_rate=0.1,        # step size shrinkage
    max_depth=6,              # tree depth
    subsample=0.8,            # use 80% of data per tree
    colsample_bytree=0.8,     # use 80% of features per tree
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'    # avoids warning
)

model.fit(X_train, y_train)

# %% Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# %% Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost (Crop Variety Prediction)")
plt.show()

# %% Feature importance plot
plt.figure(figsize=(8,5))
importances = model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - XGBoost")
plt.show()
