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