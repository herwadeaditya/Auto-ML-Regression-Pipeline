import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


# ==============================
# HEADER
# ==============================

print("\n" + "="*65)
print("🚀 AUTOMATED MACHINE LEARNING REGRESSION PIPELINE")
print("="*65)


# ==============================
# LOAD DATASET
# ==============================

file_path = input("\n📂 Enter CSV file path: ")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print("❌ Error loading dataset:", e)
    sys.exit()

print("\n✅ Dataset Loaded Successfully")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==============================
# REMOVE ID COLUMNS
# ==============================

for col in df.columns:
    if df[col].nunique() == len(df):
        df.drop(columns=[col], inplace=True)

print("🧹 ID-like columns removed")


# ==============================
# HANDLE MISSING VALUES
# ==============================

df = df.dropna(thresh=len(df)*0.6, axis=1)

for col in df.select_dtypes(include=["int64","float64"]):
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object","string"]):
    df[col] = df[col].fillna(df[col].mode()[0])

print("🔧 Missing values handled")


# ==============================
# ENCODE CATEGORICAL VARIABLES
# ==============================

df = pd.get_dummies(df, drop_first=True)

print("🔄 Categorical variables encoded")


# ==============================
# TARGET COLUMN
# ==============================

print("\n📊 Total Features:", len(df.columns))
print("First 20 Columns:\n")
print(list(df.columns[:20]))

target = input("\n🎯 Enter target column: ")

if target not in df.columns:
    print("❌ Invalid target column")
    sys.exit()


# ==============================
# FEATURE SELECTION
# ==============================

corr = df.corr()[target].abs()
strong = corr[corr > 0.1].index

df = df[strong]

print("📉 Feature selection completed")


# ==============================
# SPLIT DATA
# ==============================

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✂ Dataset split into training and testing sets")


# ==============================
# MODEL PIPELINES
# ==============================

models = {

    "Linear Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

    "Ridge Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10))
        ]),

    "Lasso Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=10000))
        ]),

    "ElasticNet Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.8, max_iter=10000))
        ])
}


# ==============================
# TRAIN MODELS
# ==============================

print("\n🤖 Training Models...\n")

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    results[name] = {
        "R2 Score":round(r2,4),
        "MAE":round(mae,2),
        "MSE":round(mse,2),
        "RMSE":round(rmse,2)
    }


# ==============================
# MODEL LEADERBOARD
# ==============================

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="R2 Score",ascending=False)

print("\n" + "="*65)
print("🏆 MODEL PERFORMANCE LEADERBOARD")
print("="*65)

print(results_df.to_string())


# ==============================
# BEST MODEL
# ==============================

best_model_name = results_df.index[0]
best_model = models[best_model_name]

print("\n⭐ Best Model Selected:", best_model_name)


# ==============================
# SAVE BEST MODEL
# ==============================

joblib.dump(best_model, "best_model.pkl")

print("💾 Best model saved as: best_model.pkl")


# ==============================
# ACTUAL VS PREDICTED PLOT
# ==============================

preds = best_model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, preds)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()


# ==============================
# MODEL COMPARISON CHART
# ==============================

plt.figure(figsize=(8,5))

results_df["R2 Score"].plot(kind="bar")

plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.xlabel("Models")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==============================
# FEATURE IMPORTANCE
# ==============================

print("\n📊 Top Important Features")

if hasattr(best_model.named_steps["model"], "coef_"):

    importance = pd.Series(
        best_model.named_steps["model"].coef_,
        index=X.columns
    ).abs().sort_values(ascending=False)

    print(importance.head(10))


# ==============================
# COMPLETION MESSAGE
# ==============================

print("\n" + "="*65)
print("✅ Pipeline Execution Completed Successfully")
print("="*65)
