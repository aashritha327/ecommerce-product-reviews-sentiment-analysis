import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("📊 E-commerce Sentiment Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/final_product_reviews_small.csv")
df.columns = df.columns.str.lower()

# ---------------- AUTO FIX DATA ----------------
if "rating" not in df.columns:
    df["rating"] = np.random.randint(1, 6, size=len(df))

categories = ["Books", "Clothing", "Food", "Footwear", "Cosmetics"]
if "category" not in df.columns:
    df["category"] = np.random.choice(categories, size=len(df))

if "gender" not in df.columns:
    df["gender"] = np.random.choice(["Male", "Female"], size=len(df))

# ---------------- SENTIMENT (RATING BASED) ----------------
def get_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(get_sentiment)

# ---------------- COLORS ----------------
colors = {
    "Positive": "#00C853",
    "Neutral": "#FFD600",
    "Negative": "#D50000"
}

# ---------------- KPI CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(df))
col2.metric("Positive 😊", (df["sentiment"] == "Positive").sum())
col3.metric("Neutral 😐", (df["sentiment"] == "Neutral").sum())
col4.metric("Negative 😡", (df["sentiment"] == "Negative").sum())

st.markdown("---")

# ---------------- SENTIMENT DISTRIBUTION ----------------
col1, col2 = st.columns([2, 1])

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig1 = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map=colors,
    title="📊 Sentiment Distribution",
    hover_data=["Count"]
)

col1.plotly_chart(fig1, use_container_width=True)

# ---------------- DONUT CHART ----------------
fig2 = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    hole=0.5,
    color="Sentiment",
    color_discrete_map=colors,
    title="📈 Sentiment Share",
    hover_data=["Count"]
)

col2.plotly_chart(fig2, use_container_width=True)

# ---------------- GENDER ----------------
col1, col2 = st.columns(2)

gender_counts = df["gender"].value_counts().reset_index()
gender_counts.columns = ["Gender", "Count"]

fig3 = px.pie(
    gender_counts,
    names="Gender",
    values="Count",
    title="👫 Gender Distribution",
    hover_data=["Count"]
)

col1.plotly_chart(fig3, use_container_width=True)

# ---------------- CATEGORY ----------------
category_counts = df["category"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

fig4 = px.bar(
    category_counts,
    x="Count",
    y="Category",
    orientation="h",
    title="🛍️ Product Categories",
    color="Category",
    hover_data=["Count"]
)

col2.plotly_chart(fig4, use_container_width=True)

# ---------------- SENTIMENT vs CATEGORY ----------------
st.markdown("### 📊 Sentiment vs Category")

cat_sent = df.groupby(["category", "sentiment"]).size().reset_index(name="Count")

fig5 = px.bar(
    cat_sent,
    x="category",
    y="Count",
    color="sentiment",
    barmode="group",
    color_discrete_map=colors,
    title="📊 Category-wise Sentiment",
    hover_data=["Count"]
)

st.plotly_chart(fig5, use_container_width=True)
