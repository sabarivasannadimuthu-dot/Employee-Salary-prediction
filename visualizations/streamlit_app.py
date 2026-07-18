"""
Interactive Streamlit dashboard for the Salary dataset.

Run with:
    pip install -r visualizations/requirements.txt
    streamlit run visualizations/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary EDA & Prediction", layout="centered")

# Load CSV (same fallback logic)
possible_paths = [
    Path("Employee-Salary-Prediction/data/Salary_dataset.csv"),
    Path("Employee-Salary-prediction/data/Salary_dataset.csv"),
    Path("data/Salary_dataset.csv"),
]
csv_path = None
for p in possible_paths:
    if p.exists():
        csv_path = p
        break
if csv_path is None:
    st.error("Salary_dataset.csv not found. Please add it to one of: " + ", ".join(str(p) for p in possible_paths))
    st.stop()

df = pd.read_csv(csv_path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.rename(columns=lambda c: c.strip())

st.title("Employee Salary — EDA & Simple Prediction")
st.write("Data source:", str(csv_path))

if st.checkbox("Show raw data"):
    st.dataframe(df)

st.sidebar.header("Plot options")
plot_type = st.sidebar.selectbox("Choose plot", ["Scatter + Regression", "Histogram (Years)", "Boxplot (Salary)", "Correlation heatmap"])
show_regression = st.sidebar.checkbox("Show regression line (for scatter)", value=True)

if plot_type == "Scatter + Regression":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="YearsExperience", y="Salary", data=df, s=70, ax=ax)
    if show_regression:
        # fit and plot
        X = df[["YearsExperience"]].values
        y = df[["Salary"]].values.ravel()
        model = LinearRegression().fit(X, y)
        x_vals = np.linspace(X.min(), X.max(), 100)[:, None]
        y_pred = model.predict(x_vals)
        ax.plot(x_vals, y_pred, color="red", linewidth=2)
        st.write(f"Model: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} * YearsExperience  (R^2={model.score(X,y):.3f})")
    st.pyplot(fig)

elif plot_type == "Histogram (Years)":
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["YearsExperience"], bins=10, kde=True, ax=ax, color="steelblue")
    st.pyplot(fig)

elif plot_type == "Boxplot (Salary)":
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df["Salary"], ax=ax, color="lightgreen")
    st.pyplot(fig)

elif plot_type == "Correlation heatmap":
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    st.pyplot(fig)

st.sidebar.header("Prediction")
st.sidebar.write("Enter years of experience to predict salary using linear regression.")
years = st.sidebar.slider("YearsExperience", float(df["YearsExperience"].min()), float(df["YearsExperience"].max()), float(df["YearsExperience"].mean()))
if st.sidebar.button("Predict"):
    X = df[["YearsExperience"]].values
    y = df["Salary"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(np.array([[years]]))[0]
    st.success(f"Predicted Salary for {years:.2f} years: ${pred:,.2f}")

st.markdown("---")
st.subheader("Summary statistics")
st.table(df.describe())
