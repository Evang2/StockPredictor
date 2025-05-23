import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
import socket
import os

from data_fetcher import fetch_stock_data, fetch_vix_data, fetch_macro_indicators
from sentiment_analyzer import (
    fetch_news_sentiment_finnhub,
    fetch_sentiment_vader,
    fetch_sentiment_bert,
)
from models.linear_model import train_linear_model
from models.prophet_model import train_prophet_model
from models.xgboost_model import train_xgboost_model
from visualizer import (
    plot_historical_prices,
    plot_linear_forecast,
    plot_prophet_forecast,
    plot_with_moving_average,
    plot_xgboost_forecast,
)
from utils import merge_all_features, add_sentiment_column

st.set_page_config(page_title="Smart Stock Predictor", layout="centered")
st.title("📈 Smart Stock Predictor")
st.markdown(
    "Predict future prices using Machine Learning + News Sentiment, Macroeconomic Indicators, and VIX (Volatility Index)."
)

# Detect if running on Streamlit Cloud
is_streamlit_cloud = os.getenv("STREAMLIT_SERVER_HEADLESS", "") == "1"

# Sidebar input controls
st.sidebar.header("🔍 Input Options")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
model_choice = st.sidebar.selectbox(
    "Choose Forecasting Model", ["Prophet", "Linear Regression", "XGBoost"]
)
compare_models = st.sidebar.checkbox("📊 Compare All Models")
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)
sentiment_model = st.sidebar.selectbox(
    "Sentiment Analysis Model", ["TextBlob", "VADER", "BERT"]
)
debug_mode = st.sidebar.checkbox("🛠️ Enable Debug Mode")

# Try fetching stock data or fallback
fallback_mode = False
try:
    df = fetch_stock_data(ticker)
    if df.empty or "Close" not in df.columns:
        raise ValueError("No valid data")
except Exception:
    if is_streamlit_cloud:
        fallback_mode = True
        st.warning(f"⚠️ Live data fetch failed. Showing demo data for AAPL instead.")
        df = pd.read_csv("fallback_aapl.csv")
        ticker = "AAPL"
    else:
        st.error(f"❌ Oops! Something went wrong with the ticker '{ticker}'. Please make sure it's a valid stock symbol.")
        st.stop()

# Fetch remaining data
vix_df = fetch_vix_data()
macro_df = fetch_macro_indicators()
raw_sentiment, headlines = fetch_news_sentiment_finnhub(ticker)

if sentiment_model == "VADER":
    sentiment_score, label = fetch_sentiment_vader(headlines)
elif sentiment_model == "BERT":
    sentiment_score = fetch_sentiment_bert(headlines)
    label = "N/A"
else:
    sentiment_score = raw_sentiment
    label = "N/A"

df = merge_all_features(df, vix_df, macro_df, sentiment_score)
df = add_sentiment_column(df, sentiment_score)

st.subheader("📈 Price with Moving Average")
st.pyplot(plot_with_moving_average(df))

st.subheader(f"📊 Historical Data: {ticker.upper()}")
st.pyplot(plot_historical_prices(df))

st.subheader("📰 Latest News Headlines")
for h in headlines:
    st.markdown(f"- {h}")
st.success(f"📊 Average Sentiment Score ({sentiment_model}): {sentiment_score:.3f}  —  Label: {label}")

if debug_mode:
    st.markdown("#### 🧪 Debug: Sentiment Analysis Details")
    st.code("Raw Headlines:", language="text")
    st.write(headlines)
    st.write("🔍 Sentiment Scores:")
    st.write({
        "TextBlob": raw_sentiment,
        "VADER": fetch_sentiment_vader(headlines)[0],
        "BERT": fetch_sentiment_bert(headlines),
    })
    st.write("📋 Merged DataFrame Preview:")
    st.write(df.tail(10))

# Forecasting section continues as in your latest logic...
# (You can paste the rest of your model comparison and plotting logic here unchanged)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Built by Evang2")
