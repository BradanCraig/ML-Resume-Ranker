import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


df = pd.read_csv("dataset.csv")

print("Original Dataset")
print(df.head())


df = df.replace(-1, np.nan)

print("\nDataset After Replacing -1 with NaN")
print(df.head())


FEATURE_COLUMNS = [
    "objective_job_similarity",
    "experience_years",
    "education_match",
    "certification_overlap",
    "matching_skills",
]

TARGET_COLUMN = "matched_score"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42,
            ),
        ),
    ]
)


model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Scores")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


regressor = model.named_steps["regressor"]

feature_importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": regressor.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False,
)

print("\nFeature Importance")
print(feature_importance)


plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["feature"],
    feature_importance["importance"],
)

plt.xticks(rotation=20)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("graphs/feature_importance.png")


plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    predictions,
)

min_val = min(min(y_test), min(predictions))
max_val = max(max(y_test), max(predictions))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.savefig("graphs/actual_vs_predictd.png")


errors = y_test - predictions

plt.figure(figsize=(10, 6))

plt.hist(
    errors,
    bins=20,
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()
plt.savefig("graphs/prediction_vs_error.png")


plt.figure(figsize=(10, 6))

plt.scatter(
    df["experience_years"],
    df["matched_score"],
)

plt.xlabel("Experience Years")
plt.ylabel("Matched Score")
plt.title("Experience vs Matched Score")

plt.tight_layout()
plt.savefig("graphs/expirence_vs_matched.png")


plt.figure(figsize=(10, 6))

plt.scatter(
    df["objective_job_similarity"],
    df["matched_score"],
)

plt.xlabel("Objective Job Similarity")
plt.ylabel("Matched Score")
plt.title("Objective Similarity vs Match Score")

plt.tight_layout()
plt.savefig("graphs/similarity_vs_matched.png")


correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

plt.imshow(correlation_matrix)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
)

plt.colorbar()

plt.title("Correlation Matrix")

plt.tight_layout()
plt.savefig("graphs/correlation_matrix.png")


plt.figure(figsize=(10, 6))

plt.hist(
    df["matched_score"],
    bins=20,
)

plt.xlabel("Matched Score")
plt.ylabel("Frequency")
plt.title("Distribution of Matched Scores")

plt.tight_layout()
plt.savefig("graphs/districution_matched.png")


plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7,
)

# perfect prediction line
min_val = min(min(y_test), min(predictions))
max_val = max(max(y_test), max(predictions))

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
)

plt.xlabel("Actual Match Score")
plt.ylabel("Predicted Match Score")
plt.title("Actual vs Predicted Match Scores")

plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png")

df["predicted_match_score"] = model.predict(X)

top_5_scores = df["predicted_match_score"].sort_values(ascending=False).head(5)

print(top_5_scores)
