import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Analytics", page_icon="🛡️", layout="wide")

DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", "data/creditcard.csv"))

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Hour"] = (df["Time"] // 3600).astype(int)
    df["Transaction Type"] = np.where(df["Class"].eq(1), "Fraud", "Legitimate")
    return df

st.title("🛡️ Credit Card Fraud Analytics")
st.caption("Interactive EDA dashboard — class imbalance, transaction amount, time patterns and anomaly-oriented exploration.")

if not DATA_PATH.exists():
    st.warning(f"Dataset not found: {DATA_PATH.resolve()}")
    st.markdown("Add **creditcard.csv** to the `data/` folder, then restart the app.")
    st.stop()

df = load_data(DATA_PATH)

with st.sidebar:
    st.header("Dashboard Filters")
    classes = st.multiselect("Transaction type", ["Legitimate", "Fraud"], ["Legitimate", "Fraud"])
    max_hour = int(df["Hour"].max())
    hour_range = st.slider("Hour index", 0, max_hour, (0, max_hour))
    max_amount = float(df["Amount"].max())
    amount_range = st.slider("Amount range", 0.0, max_amount, (0.0, max_amount))

filtered = df[
    df["Transaction Type"].isin(classes)
    & df["Hour"].between(hour_range[0], hour_range[1])
    & df["Amount"].between(amount_range[0], amount_range[1])
]

total = len(filtered)
fraud_count = int(filtered["Class"].sum())
fraud_rate = fraud_count / total * 100 if total else 0
avg_amount = filtered["Amount"].mean() if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{total:,}")
c2.metric("Fraud Cases", f"{fraud_count:,}")
c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
c4.metric("Average Amount", f"${avg_amount:,.2f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    counts = filtered["Transaction Type"].value_counts().rename_axis("Type").reset_index(name="Count")
    fig = px.bar(counts, x="Type", y="Count", color="Type", text_auto=True,
                 title="Legitimate vs Fraud Transactions")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    hourly = filtered.groupby(["Hour", "Transaction Type"]).size().reset_index(name="Count")
    fig = px.line(hourly, x="Hour", y="Count", color="Transaction Type", markers=True,
                  title="Transaction Activity by Hour")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.box(filtered, x="Transaction Type", y="Amount", color="Transaction Type",
                 title="Transaction Amount Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fraud_hour = filtered.loc[filtered["Class"].eq(1)].groupby("Hour").size().reset_index(name="Fraud Cases")
    fig = px.bar(fraud_hour, x="Hour", y="Fraud Cases", title="Fraud Cases by Hour")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Preview")
st.dataframe(
    filtered[["Time", "Hour", "Amount", "Class", "Transaction Type"]].head(100),
    use_container_width=True
)

st.info("Interpret fraud rate together with absolute fraud counts because the dataset is highly imbalanced. Anomaly detection and fraud classification are not the same task.")
