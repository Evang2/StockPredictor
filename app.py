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

# Streamlit page config
st.set_page_config(page_title="Smart Stock Predictor", layout="centered")
st.title("📈 Smart Stock Predictor")
st.markdown(
    "Predict future prices using Machine Learning + News Sentiment, Macroeconomic Indicators, and VIX (Volatility Index)."
)

# Detect Streamlit Cloud environment
is_streamlit_cloud = os.getenv("STREAMLIT_SERVER_HEADLESS", "") == "1"

# Sidebar controls
st.sidebar.header("🔍 Input Options")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["Prophet", "Linear Regression", "XGBoost"])
compare_models = st.sidebar.checkbox("📊 Compare All Models")
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)
sentiment_model = st.sidebar.selectbox("Sentiment Analysis Model", ["TextBlob", "VADER", "BERT"])
debug_mode = st.sidebar.checkbox("🛠️ Enable Debug Mode")

# === Load Stock Data with Fallback ===
fallback_mode = False
try:
    df = fetch_stock_data(ticker)
    if df.empty or "Close" not in df.columns:
        raise ValueError("Invalid or empty stock data")
except Exception:
    if is_streamlit_cloud:
        try:
            df = pd.read_csv("fallback_aapl.csv")
            if df.empty or "Close" not in df.columns:
                raise ValueError("Fallback data is invalid.")
            ticker = "AAPL"
            fallback_mode = True
            st.warning("⚠️ Live stock data unavailable. Loaded fallback demo data (AAPL).")
        except Exception:
            st.error("❌ Could not load fallback demo data. Please try again later.")
            st.stop()
    else:
        st.error(f"❌ Oops! Something went wrong with the ticker '{ticker}'. Please make sure it's a valid stock symbol.")
        st.stop()

# === Fetch Other Data ===
try:
    vix_df = fetch_vix_data()
except Exception:
    st.warning("⚠️ Failed to fetch VIX data. Continuing without it.")
    vix_df = pd.DataFrame()

try:
    macro_df = fetch_macro_indicators()
except Exception:
    st.warning("⚠️ Failed to fetch macroeconomic indicators. Continuing without them.")
    macro_df = pd.DataFrame()

try:
    raw_sentiment, headlines = fetch_news_sentiment_finnhub(ticker)
except Exception:
    st.warning("⚠️ Failed to fetch news sentiment. Using default.")
    raw_sentiment = 0.0
    headlines = ["No headlines available"]

# === Sentiment Model Handling ===
if sentiment_model == "VADER":
    sentiment_score, label = fetch_sentiment_vader(headlines)
elif sentiment_model == "BERT":
    sentiment_score = fetch_sentiment_bert(headlines)
    label = "N/A"
else:
    sentiment_score = raw_sentiment
    label = "N/A"

# === Merge Final Dataset ===
df = merge_all_features(df, vix_df, macro_df, sentiment_score)
df = add_sentiment_column(df, sentiment_score)

# === Visualization ===
st.subheader("📈 Price with Moving Average")
st.pyplot(plot_with_moving_average(df))

st.subheader(f"📊 Historical Data: {ticker.upper()}")
st.pyplot(plot_historical_prices(df))

st.subheader("📰 Latest News Headlines")
for h in headlines:
    st.markdown(f"- {h}")
st.success(f"📊 Avg Sentiment Score ({sentiment_model}): `{sentiment_score:.3f}` — Label: {label}")

if debug_mode:
    st.markdown("#### 🧪 Debug Panel")
    st.write("Headlines:", headlines)
    st.write("Sentiments:", {
        "TextBlob": raw_sentiment,
        "VADER": fetch_sentiment_vader(headlines)[0],
        "BERT": fetch_sentiment_bert(headlines)
    })
    st.dataframe(df.tail(10))

# === Forecast Section ===
st.subheader("🔮 Forecast")

try:
    if compare_models:
        forecast_linear = train_linear_model(df)
        forecast_prophet = train_prophet_model(df)
        forecast_xgb = train_xgboost_model(df)

        merged = pd.merge(
            forecast_linear.rename(columns={"PredictedClose": "Linear"}),
            forecast_prophet[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Prophet"}),
            on="Date", how="inner"
        )
        merged = pd.merge(
            merged,
            forecast_xgb.rename(columns={"PredictedClose": "XGBoost"}),
            on="Date", how="inner"
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["Date"], df["Close"], label="Historical", color="black")
        ax.plot(merged["Date"], merged["Linear"], label="Linear", linestyle="--", color="orange")
        ax.plot(merged["Date"], merged["Prophet"], label="Prophet", linestyle="--", color="green")
        ax.plot(merged["Date"], merged["XGBoost"], label="XGBoost", linestyle="--", color="blue")
        ax.set_title(f"{ticker.upper()} Forecast Comparison")
        ax.legend()
        st.pyplot(fig)

        st.dataframe(merged.tail(forecast_days))
        st.download_button("📥 Download Comparison CSV", data=merged.to_csv(index=False), file_name="comparison.csv", mime="text/csv")

        if debug_mode:
            true = df["Close"].tail(forecast_days).values
            scores = {
                "Linear Regression": (mean_absolute_percentage_error(true, merged["Linear"][:forecast_days]) * 100,
                                      np.sqrt(mean_squared_error(true, merged["Linear"][:forecast_days]))),
                "Prophet": (mean_absolute_percentage_error(true, merged["Prophet"][:forecast_days]) * 100,
                            np.sqrt(mean_squared_error(true, merged["Prophet"][:forecast_days]))),
                "XGBoost": (mean_absolute_percentage_error(true, merged["XGBoost"][:forecast_days]) * 100,
                            np.sqrt(mean_squared_error(true, merged["XGBoost"][:forecast_days])))
            }

            best_model = min(scores.items(), key=lambda x: x[1][0])[0]
            st.success(f"🏆 Best Model: {best_model} (MAPE: {scores[best_model][0]:.2f}%)")

            st.markdown("### 📊 Model Performance Metrics")
            st.write(pd.DataFrame(scores, index=["MAPE", "RMSE"]).T)

            fig_score = go.Figure()
            for i, metric in enumerate(["MAPE", "RMSE"]):
                fig_score.add_trace(go.Bar(
                    name=metric,
                    x=list(scores.keys()),
                    y=[s[i] for s in scores.values()]
                ))
            fig_score.update_layout(title="Model Scores", barmode="group")
            st.plotly_chart(fig_score, use_container_width=True)

    else:
        if model_choice == "Linear Regression":
            forecast_df = train_linear_model(df)
            st.pyplot(plot_linear_forecast(df, forecast_df))
        elif model_choice == "Prophet":
            forecast_df = train_prophet_model(df)
            st.pyplot(plot_prophet_forecast(df, forecast_df))
        else:
            forecast_df = train_xgboost_model(df)
            st.pyplot(plot_xgboost_forecast(df, forecast_df))

        st.dataframe(forecast_df.tail(forecast_days))
        st.download_button("📥 Download Forecast CSV", data=forecast_df.to_csv(index=False), file_name=f"{ticker}_{model_choice}.csv", mime="text/csv")

except Exception as e:
    st.error("❌ Forecasting failed. Try adjusting your input or check the logs.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Built by Evang2")
