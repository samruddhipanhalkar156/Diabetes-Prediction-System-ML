# ============================
# main.py - CDC Diabetes Health Indicators
# ============================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

# =====================
# Folder Setup
# =====================
EDA_DIR = "EDA Results"
MODELS_DIR = "Models"
RESULTS_DIR = "Model Results"
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# Load Dataset
# =====================
print("📥 Loading CDC Diabetes Health Indicators dataset...")
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=891)

X = dataset.data.features
y = dataset.data.targets

# Drop unwanted columns
drop_cols = [col for col in ["ID", "Education", "Income"] if col in X.columns]
if drop_cols:
    print(f"🗑️ Dropping columns: {drop_cols}")
    X = X.drop(columns=drop_cols)

# Ensure correct target column
if "Diabetes_binary" in y.columns:
    y = y["Diabetes_binary"]

# =====================
# Balanced Sampling (Max 5000 per class)
# =====================
print("⚖️ Sampling 5000 per class for balance...")
df = pd.concat([X, y], axis=1)

class_0 = df[df["Diabetes_binary"] == 0]
class_1 = df[df["Diabetes_binary"] == 1]

n_samples = 5000
class_0_sample = resample(class_0, replace=False, n_samples=n_samples, random_state=42)
class_1_sample = resample(class_1, replace=False, n_samples=n_samples, random_state=42)

df_balanced = pd.concat([class_0_sample, class_1_sample])

X = df_balanced.drop(columns=["Diabetes_binary"])
y = df_balanced["Diabetes_binary"]

print(f"✅ Dataset size after sampling: {X.shape}, Class balance: {y.value_counts().to_dict()}")

# =====================
# EDA
# =====================
print("📊 Generating EDA plots...")

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Balanced Class Distribution: Diabetes (0=No, 1=Yes)")
plt.savefig(os.path.join(EDA_DIR, f"class_distribution_{dt.now().strftime('%y_%b_%d_%H_%M')}.png"))
plt.close()

plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(EDA_DIR, f"heatmap_{dt.now().strftime('%y_%b_%d_%H_%M')}.png"))
plt.close()

if "BMI" in X.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=y, y=X["BMI"])
    plt.title("BMI vs Diabetes")
    plt.savefig(os.path.join(EDA_DIR, f"bmi_boxplot_{dt.now().strftime('%y_%b_%d_%H_%M')}.png"))
    plt.close()

# =====================
# Scaling
# =====================
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# =====================
# Train-Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# Models & Parameters (Simplified SVM for speed)
# =====================
print("🤖 Starting Model Training...")
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 5, 10], "solver": ["newton-cg", "lbfgs"], "multi_class": ["ovr"]}
    ),
    "SVM": (
        SVC(probability=True),
        {"C": [0.1, 1], "kernel": ["linear"], "gamma": ["scale"]}  # Simplified to avoid CPU overload
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {"n_estimators": [100], "max_depth": [5, 10, None]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1], "subsample": [0.8, 1.0]}
    )
}

results = []

# =====================
# Training Function
# =====================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    for name, (model, params) in models.items():
        print(f"🔍 Training {name}...")
        grid = GridSearchCV(
            model,
            params,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # reduce folds for speed
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Save model
        filename = f"{name}_diabetes.pkl"
        with open(os.path.join(MODELS_DIR, filename), "wb") as f:
            pickle.dump(best_model, f)

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        y_train_prob = best_model.predict_proba(X_train)[:,1]
        y_test_prob = best_model.predict_proba(X_test)[:,1]

        auc_train = roc_auc_score(y_train, y_train_prob)
        auc_test = roc_auc_score(y_test, y_test_prob)

        for split, y_true, y_pred, auc in [
            ("Train", y_train, y_train_pred, auc_train),
            ("Test", y_test, y_test_pred, auc_test)
        ]:
            results.append({
                "Model": name,
                "Split": split,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision_weighted": precision_score(y_true, y_pred, average="weighted"),
                "Recall_weighted": recall_score(y_true, y_pred, average="weighted"),
                "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
                "Precision_macro": precision_score(y_true, y_pred, average="macro"),
                "Recall_macro": recall_score(y_true, y_pred, average="macro"),
                "F1_macro": f1_score(y_true, y_pred, average="macro"),
                "AUC ROC": auc
            })

# =====================
# Run Training
# =====================
train_and_evaluate(X_train, X_test, y_train, y_test)

# =====================
# Save Results
# =====================
results_df = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, f"model_results_{dt.now().strftime('%y_%b_%d_%H_%M')}.xlsx")
results_df.to_excel(results_file, index=False)

print("✅ Pipeline completed! Models and results are saved.")
