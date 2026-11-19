# ==========================================================
# KARNATAKA ELECTRICITY DEMAND PREDICTION
# LINEAR REGRESSION MODEL
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# STEP 1: CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs("output", exist_ok=True)

print("\n====================================")
print(" LINEAR REGRESSION MODEL")
print("====================================")


# ==========================================================
# STEP 2: LOAD CLEANED DATASET
# ==========================================================

df = pd.read_csv(
    "output/cleaned_electricity_data.csv"
)

print("\nDataset Loaded Successfully!")

print("Original Dataset Shape:", df.shape)


# ==========================================================
# STEP 3: DATA CLEANING
# ==========================================================

# Check required columns

required_columns = [
    "TIME STAMP",
    "state_demand"
]

for col in required_columns:

    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )


# Convert timestamp

df["TIME STAMP"] = pd.to_datetime(
    df["TIME STAMP"],
    errors="coerce"
)


# Convert target to numeric

df["state_demand"] = pd.to_numeric(
    df["state_demand"],
    errors="coerce"
)


# Remove invalid timestamps and demand values

df = df.dropna(
    subset=["TIME STAMP", "state_demand"]
)


# Remove invalid sentinel values

df = df[
    df["state_demand"] != -99999
]


# Sort dataset chronologically

df = df.sort_values(
    "TIME STAMP"
).reset_index(drop=True)


print("\nData Cleaning Completed!")

print("Cleaned Dataset Shape:", df.shape)


# ==========================================================
# STEP 4: FEATURE ENGINEERING
# ==========================================================

# Extract time-based features

df["hour"] = df["TIME STAMP"].dt.hour

df["day"] = df["TIME STAMP"].dt.day

df["month"] = df["TIME STAMP"].dt.month

df["day_of_week"] = (
    df["TIME STAMP"].dt.dayofweek
)


# Historical electricity demand features

# Previous observation

df["demand_lag_1"] = (
    df["state_demand"].shift(1)
)


# Demand from two observations earlier

df["demand_lag_2"] = (
    df["state_demand"].shift(2)
)


# Demand from four observations earlier

df["demand_lag_4"] = (
    df["state_demand"].shift(4)
)


# Average demand of previous four observations

df["demand_rolling_mean"] = (
    df["state_demand"]
    .shift(1)
    .rolling(window=4)
    .mean()
)


# Remove rows with missing lag values

df = df.dropna().reset_index(drop=True)


print("\nFeature Engineering Completed!")


# ==========================================================
# STEP 5: DEFINE INPUT FEATURES AND TARGET
# ==========================================================

features = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_4",
    "demand_rolling_mean"
]


X = df[features]

y = df["state_demand"]


print("\nINPUT FEATURES:")

print(X.columns.tolist())


print("\nTARGET VARIABLE:")

print(y.name)


# Check direct target leakage

if y.name in X.columns:

    raise ValueError(
        "Target leakage detected!"
    )

else:

    print("\nNo direct target leakage detected!")


# ==========================================================
# STEP 6: CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================================

# Do not use random train_test_split.
# Earlier observations are used for training.
# Later observations are used for testing.

split_index = int(len(df) * 0.80)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


print("\nTRAIN-TEST SPLIT COMPLETED!")

print("Training Data:", X_train.shape)

print("Testing Data:", X_test.shape)


print("\nTraining Period:")

print(
    df["TIME STAMP"].iloc[0],
    "to",
    df["TIME STAMP"].iloc[split_index - 1]
)


print("\nTesting Period:")

print(
    df["TIME STAMP"].iloc[split_index],
    "to",
    df["TIME STAMP"].iloc[-1]
)


# Verify chronological separation

if (
    df["TIME STAMP"].iloc[split_index - 1]
    >=
    df["TIME STAMP"].iloc[split_index]
):

    raise ValueError(
        "Training and testing periods overlap!"
    )


# ==========================================================
# STEP 7: BUILD LINEAR REGRESSION PIPELINE
# ==========================================================

# StandardScaler is fitted only on training data
# because it is inside the Pipeline.

model = Pipeline([

    ("scaler", StandardScaler()),

    ("regressor", LinearRegression())

])


# ==========================================================
# STEP 8: TRAIN THE MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


print("\nLinear Regression Model Trained Successfully!")


# ==========================================================
# STEP 9: PREDICT ELECTRICITY DEMAND
# ==========================================================

y_pred = model.predict(X_test)


print("\nElectricity Demand Prediction Completed!")


# ==========================================================
# STEP 10: MODEL PERFORMANCE EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


mse = mean_squared_error(
    y_test,
    y_pred
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_test,
    y_pred
)


print("\n====================================")
print(" LINEAR REGRESSION PERFORMANCE")
print("====================================")


print("Mean Absolute Error (MAE):", round(mae, 3))

print("Mean Squared Error (MSE):", round(mse, 3))

print("Root Mean Squared Error (RMSE):", round(rmse, 3))

print("R2 Score:", round(r2, 4))


# ==========================================================
# STEP 11: SAVE PREDICTION RESULTS
# ==========================================================

predictions = pd.DataFrame({

    "TIME STAMP":
        df["TIME STAMP"].iloc[split_index:].values,

    "Actual Demand":
        y_test.values,

    "Predicted Demand":
        y_pred,

    "Prediction Error":
        y_test.values - y_pred

})


predictions.to_csv(
    "output/linear_regression_predictions.csv",
    index=False
)


print("\nPrediction Results Saved Successfully!")


# ==========================================================
# STEP 12: SAVE PERFORMANCE METRICS
# ==========================================================

metrics = pd.DataFrame({

    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],

    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]

})


metrics.to_csv(
    "output/linear_regression_metrics.csv",
    index=False
)


print("Performance Metrics Saved Successfully!")


# ==========================================================
# STEP 13: ACTUAL VS PREDICTED GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))


plt.scatter(

    y_test,

    y_pred,

    alpha=0.6,

    label="Predicted Values"

)


plt.plot(

    [
        y_test.min(),
        y_test.max()
    ],

    [
        y_test.min(),
        y_test.max()
    ],

    color="red",

    linestyle="--",

    label="Perfect Prediction"

)


plt.xlabel("Actual Electricity Demand")

plt.ylabel("Predicted Electricity Demand")


plt.title(
    "Linear Regression: Actual vs Predicted"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "output/linear_regression_actual_vs_predicted.png",

    dpi=300

)


plt.close()


# ==========================================================
# STEP 14: ELECTRICITY DEMAND PREDICTION GRAPH
# ==========================================================

plt.figure(figsize=(12, 6))


plt.plot(

    predictions["TIME STAMP"],

    predictions["Actual Demand"],

    label="Actual Demand"

)


plt.plot(

    predictions["TIME STAMP"],

    predictions["Predicted Demand"],

    label="Predicted Demand"

)


plt.xlabel("Date and Time")

plt.ylabel("Electricity Demand")


plt.title(
    "Karnataka Electricity Demand Prediction - Linear Regression"
)


plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()


plt.savefig(

    "output/linear_regression_graph.png",

    dpi=300

)


plt.close()


# ==========================================================
# STEP 15: SAVE MODEL COEFFICIENTS
# ==========================================================

coefficients = pd.DataFrame({

    "Feature": features,

    "Coefficient":
        model.named_steps["regressor"].coef_

})


coefficients.to_csv(

    "output/linear_regression_coefficients.csv",

    index=False

)


print("\nModel Coefficients Saved Successfully!")


# ==========================================================
# STEP 16: FINAL OUTPUT
# ==========================================================

print("\n====================================")

print("LINEAR REGRESSION IMPLEMENTATION COMPLETED!")

print("====================================")


print("\nGENERATED FILES:")


print("1. linear_regression_predictions.csv")

print("2. linear_regression_metrics.csv")

print("3. linear_regression_actual_vs_predicted.png")

print("4. linear_regression_graph.png")

print("5. linear_regression_coefficients.csv")


print("\nAll results saved in output folder!")

print("\n====================================")