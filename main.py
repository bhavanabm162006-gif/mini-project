# =====================================================
# KARNATAKA STATE ELECTRICITY DEMAND AND GENERATION
# DATA CLEANING, STATISTICAL ANALYSIS AND EDA
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# STEP 1: LOAD DATASET
# -----------------------------------------------------

file_path = "data/Karnataka_SLDC 2 (2).csv"

df = pd.read_csv(file_path)

print("\nOriginal Dataset Loaded Successfully!")
print("Original Dataset Shape:", df.shape)

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()


# -----------------------------------------------------
# STEP 2: SELECT ELECTRICITY-RELATED COLUMNS
# -----------------------------------------------------

electricity_columns = [
    "current_frequency",
    "state_demand",
    "state_generation",
    "cgs",
    "ncep_generation",
    "bescom",
    "hescom",
    "cesc",
    "gescom",
    "mescom",
    "pavagada_kspdcl",
    "TIME STAMP"
]

# Check required columns
missing_columns = [
    col for col in electricity_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# Exclude all weather columns
df = df[electricity_columns].copy()

print("\nWeather columns excluded successfully!")
print("Electricity Dataset Shape:", df.shape)


# -----------------------------------------------------
# STEP 3: DATA CLEANING
# -----------------------------------------------------

# Remove duplicate records
df = df.drop_duplicates()

# Convert timestamp into datetime format
df["TIME STAMP"] = pd.to_datetime(
    df["TIME STAMP"],
    errors="coerce",
    dayfirst=True
)

# Remove invalid timestamps
df = df.dropna(subset=["TIME STAMP"])

# Sort records by timestamp
df = df.sort_values("TIME STAMP")

# Convert electricity columns into numeric format
numeric_columns = [
    "current_frequency",
    "state_demand",
    "state_generation",
    "cgs",
    "ncep_generation",
    "bescom",
    "hescom",
    "cesc",
    "gescom",
    "mescom",
    "pavagada_kspdcl"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Replace invalid sentinel values
df.replace(-99999, np.nan, inplace=True)

# Remove negative electricity demand
df.loc[
    df["state_demand"] < 0,
    "state_demand"
] = np.nan

# Remove negative electricity generation
df.loc[
    df["state_generation"] < 0,
    "state_generation"
] = np.nan

# Fill missing numerical values
df[numeric_columns] = df[numeric_columns].interpolate(
    method="linear",
    limit_direction="both"
)

# Remove remaining rows with missing numerical values
df = df.dropna(subset=numeric_columns)

# Reset index
df = df.reset_index(drop=True)

print("\nDATA CLEANING COMPLETED!")
print("Cleaned Dataset Shape:", df.shape)


# -----------------------------------------------------
# STEP 4: CREATE OUTPUT FOLDER
# -----------------------------------------------------

os.makedirs("output", exist_ok=True)


# -----------------------------------------------------
# STEP 5: SAVE CLEANED DATASET
# -----------------------------------------------------

df.to_csv(
    "output/cleaned_electricity_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# -----------------------------------------------------
# STEP 6: CHECK MISSING VALUES
# -----------------------------------------------------

print("\nMISSING VALUES AFTER CLEANING")

print(df.isnull().sum())


# -----------------------------------------------------
# STEP 7: STATISTICAL ANALYSIS
# -----------------------------------------------------

print("\nSTATISTICAL ANALYSIS")

statistics = df[numeric_columns].describe()

print(statistics)

statistics.to_csv(
    "output/statistical_analysis.csv"
)


# -----------------------------------------------------
# STEP 8: ELECTRICITY DEMAND GRAPH
# -----------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["TIME STAMP"],
    df["state_demand"]
)

plt.title("Karnataka State Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Electricity Demand")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/demand_graph.png"
)

plt.show()


# -----------------------------------------------------
# STEP 9: ELECTRICITY GENERATION GRAPH
# -----------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["TIME STAMP"],
    df["state_generation"]
)

plt.title("Karnataka State Electricity Generation")
plt.xlabel("Date")
plt.ylabel("Electricity Generation")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/generation_graph.png"
)

plt.show()


# -----------------------------------------------------
# STEP 10: DEMAND VS GENERATION GRAPH
# -----------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["TIME STAMP"],
    df["state_demand"],
    label="Electricity Demand"
)

plt.plot(
    df["TIME STAMP"],
    df["state_generation"],
    label="Electricity Generation"
)

plt.title("Electricity Demand vs Generation")

plt.xlabel("Date")
plt.ylabel("Electricity")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/demand_vs_generation.png"
)

plt.show()


# -----------------------------------------------------
# STEP 11: BESCOM ELECTRICITY DEMAND GRAPH
# -----------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["TIME STAMP"],
    df["bescom"]
)

plt.title("BESCOM Electricity Demand")

plt.xlabel("Date")
plt.ylabel("Electricity")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/bescom_demand_graph.png"
)

plt.show()


# -----------------------------------------------------
# STEP 12: CORRELATION ANALYSIS
# -----------------------------------------------------

correlation = df[numeric_columns].corr()

print("\nCORRELATION ANALYSIS")

print(correlation)

correlation.to_csv(
    "output/correlation.csv"
)

# Correlation Heatmap

plt.figure(figsize=(12, 8))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=90
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Electricity Demand and Generation Correlation")

plt.tight_layout()

plt.savefig(
    "output/correlation_heatmap.png"
)

plt.show()


# -----------------------------------------------------
# STEP 13: FINAL RESULTS
# -----------------------------------------------------

print("\n===================================")
print("PROJECT IMPLEMENTATION COMPLETED")
print("===================================")

print("\nFinal Dataset Shape:", df.shape)

print("\nGenerated Files:")
print("1. cleaned_electricity_data.csv")
print("2. statistical_analysis.csv")
print("3. correlation.csv")
print("4. demand_graph.png")
print("5. generation_graph.png")
print("6. demand_vs_generation.png")
print("7. bescom_demand_graph.png")
print("8. correlation_heatmap.png")

print("\nAll results saved in the output folder!")
# ==========================================
# PEAK ELECTRICITY DEMAND ANALYSIS
# ==========================================

print("\nPEAK ELECTRICITY DEMAND ANALYSIS")

# Convert timestamp into datetime format
df["TIME STAMP"] = pd.to_datetime(
    df["TIME STAMP"],
    errors="coerce"
)

# Remove invalid values
demand_df = df.dropna(
    subset=["TIME STAMP", "state_demand"]
).copy()

# Sort data by timestamp
demand_df = demand_df.sort_values("TIME STAMP")

# Maximum demand
peak_demand = demand_df["state_demand"].max()

# Minimum demand
minimum_demand = demand_df["state_demand"].min()

# Average demand
average_demand = demand_df["state_demand"].mean()

# Find peak demand date and time
peak_row = demand_df.loc[
    demand_df["state_demand"].idxmax()
]

print("\nMaximum Electricity Demand:", peak_demand)

print("Minimum Electricity Demand:", minimum_demand)

print("Average Electricity Demand:", average_demand)

print("Peak Demand Date and Time:",
      peak_row["TIME STAMP"])

# Save peak demand analysis
peak_results = pd.DataFrame({
    "Parameter": [
        "Maximum Demand",
        "Minimum Demand",
        "Average Demand",
        "Peak Demand Date"
    ],
    "Value": [
        peak_demand,
        minimum_demand,
        average_demand,
        str(peak_row["TIME STAMP"])
    ]
})

peak_results.to_csv(
    "output/peak_demand_analysis.csv",
    index=False
)

# Plot peak demand
plt.figure(figsize=(12, 5))

plt.plot(
    demand_df["TIME STAMP"],
    demand_df["state_demand"]
)

# Highlight peak electricity demand
peak_index = demand_df["state_demand"].idxmax()

plt.scatter(
    demand_df.loc[peak_index, "TIME STAMP"],
    demand_df.loc[peak_index, "state_demand"],
    color="red",
    s=100,
    label="Peak Demand"
)

plt.legend()
plt.title("Karnataka Peak Electricity Demand Analysis")
plt.xlabel("Date and Time")
plt.ylabel("Electricity Demand")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("output/peak_demand_analysis.png")

plt.show()

print("\nPeak Demand Analysis Completed!")
# =====================================================
# STEP 15: RANDOM FOREST REGRESSION
# =====================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\nRANDOM FOREST REGRESSION ANALYSIS")

# STEP 1: Select input and output variables

X = df[
    ["state_generation", "current_frequency"]
]

y = df["state_demand"]

# STEP 2: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# STEP 3: Create Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# STEP 4: Train model

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Training Completed!")

# STEP 5: Predict electricity demand

rf_predictions = rf_model.predict(X_test)

# STEP 6: Model evaluation

mae = mean_absolute_error(
    y_test, rf_predictions
)

mse = mean_squared_error(
    y_test, rf_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test, rf_predictions
)

print("\nRANDOM FOREST MODEL PERFORMANCE")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# STEP 7: Save prediction results

rf_results = pd.DataFrame({
    "Actual Demand": y_test.values,
    "Predicted Demand": rf_predictions
})

rf_results.to_csv(
    "output/random_forest_predictions.csv",
    index=False
)

# STEP 8: Actual vs Predicted Graph

plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values[:100],
    label="Actual Demand"
)

plt.plot(
    rf_predictions[:100],
    label="Predicted Demand"
)

plt.title("Random Forest Regression - Electricity Demand")

plt.xlabel("Test Samples")
plt.ylabel("Electricity Demand")

plt.legend()
plt.tight_layout()

plt.savefig(
    "output/random_forest_regression.png"
)

plt.show()

# STEP 9: Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print("\nFEATURE IMPORTANCE")
print(importance)

importance.to_csv(
    "output/random_forest_feature_importance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    "output/random_forest_feature_importance.png"
)

plt.show()

print("\nRANDOM FOREST REGRESSION COMPLETED!")

# =====================================================
# STEP 14: LINEAR REGRESSION VS RANDOM FOREST COMPARISON
# =====================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("\n======================================")
print("LINEAR REGRESSION VS RANDOM FOREST")
print("======================================")

# Select input features
X = df[[
    "state_generation",
    "current_frequency"
]]

# Select target variable
y = df["state_demand"]

# Remove missing values
model_data = pd.concat([X, y], axis=1).dropna()

X = model_data[[
    "state_generation",
    "current_frequency"
]]

y = model_data["state_demand"]

# Split data chronologically (80% training, 20% testing)
split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =====================================================
# LINEAR REGRESSION MODEL
# =====================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# =====================================================
# RANDOM FOREST REGRESSION MODEL
# =====================================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# =====================================================
# MODEL PERFORMANCE COMPARISON
# =====================================================

linear_mae = mean_absolute_error(
    y_test, linear_predictions
)

linear_mse = mean_squared_error(
    y_test, linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test, linear_predictions
)

rf_mae = mean_absolute_error(
    y_test, rf_predictions
)

rf_mse = mean_squared_error(
    y_test, rf_predictions
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test, rf_predictions
)

print("\nLINEAR REGRESSION RESULTS")

print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

print("\nRANDOM FOREST REGRESSION RESULTS")

print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# =====================================================
# SAVE MODEL COMPARISON RESULTS
# =====================================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "MSE": [
        linear_mse,
        rf_mse
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

comparison.to_csv(
    "output/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON TABLE")

print(comparison)

# =====================================================
# STEP 15: ACTUAL VS PREDICTED GRAPH
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual Demand"
)

plt.plot(
    range(len(y_test)),
    linear_predictions,
    label="Linear Regression"
)

plt.plot(
    range(len(y_test)),
    rf_predictions,
    label="Random Forest Regression"
)

plt.title("Actual vs Predicted Electricity Demand")

plt.xlabel("Test Data Records")
plt.ylabel("Electricity Demand")

plt.legend()

plt.tight_layout()

plt.savefig(
    "output/model_actual_vs_predicted.png"
)

plt.show()

# =====================================================
# STEP 16: MODEL PERFORMANCE COMPARISON GRAPH
# =====================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["R2 Score"]
)

plt.title("Linear Regression vs Random Forest R2 Score")

plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")

plt.tight_layout()

plt.savefig(
    "output/model_r2_comparison.png"
)

plt.show()

# =====================================================
# FINAL MODEL COMPARISON
# =====================================================

print("\n======================================")
print("MODEL COMPARISON COMPLETED")
print("======================================")

if rf_r2 > linear_r2:
    print("Random Forest Regression performed better.")
elif linear_r2 > rf_r2:
    print("Linear Regression performed better.")
else:
    print("Both models achieved the same R2 score.")

print("\nComparison files saved successfully!")

