"""
Simple EDA + static visualizations for the Salary dataset.

Saves generated plots to visualizations/outputs/.
This file is named Data_visualization.py as requested.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set(style="darkgrid")

# Try a few possible paths for the CSV
possible_paths = [
    Path("Employee-Salary-Prediction/data/Salary_dataset.csv"),
    Path("Employee-Salary-prediction/Employee-Salary-Prediction/data/Salary_dataset.csv"),
    Path("Employee-Salary-prediction/data/Salary_dataset.csv"),
    Path("data/Salary_dataset.csv"),
    Path("Employee-Salary-Prediction/data/Salary_dataset.csv"),
]

csv_path = None
for p in possible_paths:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    # Fall back to asking user to place it at data/Salary_dataset.csv
    raise FileNotFoundError(
        "Salary_dataset.csv not found. Please place the file at one of: "
        + ", ".join(str(p) for p in possible_paths)
    )

print("Loading data from:", csv_path)
df = pd.read_csv(csv_path)

# Drop any unnamed index columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
# Alternatively drop the first column if it's an implicit index
if df.columns[0].lower() in ("unnamed: 0", ""):
    df = df.drop(df.columns[0], axis=1)

# Ensure column names are as expected
df = df.rename(columns=lambda c: c.strip())

print("Columns:", df.columns.tolist())
print(df.describe())

# Create output folder
out_dir = Path("visualizations/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# Scatter plot + regression line (seaborn)
plt.figure(figsize=(8, 6))
sns.regplot(x="YearsExperience", y="Salary", data=df, scatter_kws={"s": 60}, line_kws={"color": "red"})
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig(out_dir / "scatter_regression.png", dpi=150)
plt.close()

# Histogram of YearsExperience
plt.figure(figsize=(8, 4))
sns.histplot(df["YearsExperience"], kde=True, bins=10, color="steelblue")
plt.title("Distribution of Years Experience")
plt.xlabel("YearsExperience")
plt.tight_layout()
plt.savefig(out_dir / "hist_years_experience.png", dpi=150)
plt.close()

# Boxplot of Salary
plt.figure(figsize=(6, 6))
sns.boxplot(y=df["Salary"], color="lightgreen")
plt.title("Salary Boxplot")
plt.tight_layout()
plt.savefig(out_dir / "boxplot_salary.png", dpi=150)
plt.close()

# Correlation heatmap (though only two numeric columns exist now)
plt.figure(figsize=(6, 5))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
plt.close()

# Fit a simple linear regression model (YearsExperience -> Salary)
X = df[["YearsExperience"]].values
y = df["Salary"].values
model = LinearRegression()
model.fit(X, y)
r2 = model.score(X, y)
coef = model.coef_[0]
intercept = model.intercept_
print(f"Linear model: Salary = {intercept:.2f} + {coef:.2f} * YearsExperience  (R^2 = {r2:.3f})")

# Plot regression line with matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(X, y, s=60, label="Data", color="tab:blue")
x_vals = np.linspace(X.min(), X.max(), 100)[:, None]
y_pred = model.predict(x_vals)
plt.plot(x_vals, y_pred, color="red", linewidth=2, label="Linear fit")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Linear regression fit")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "linear_fit.png", dpi=150)
plt.close()

# Save a CSV with predictions appended
df_pred = df.copy()
df_pred["PredictedSalary"] = model.predict(X)
df_pred.to_csv(out_dir / "salary_with_predictions.csv", index=False)

print("All plots and outputs saved to:", out_dir.resolve())
