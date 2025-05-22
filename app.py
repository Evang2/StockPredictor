import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go

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
st.markdown("Predict future prices using Machine Learning + News Sentiment, Macroeconomic Indicators, and VIX (Volatility Index).")

# Sidebar input controls
st.sidebar.header("🔍 Input Options")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["Prophet", "Linear Regression", "XGBoost"])
compare_models = st.sidebar.checkbox("📊 Compare All Models")
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)
sentiment_model = st.sidebar.selectbox("Sentiment Analysis Model", ["TextBlob", "VADER", "BERT"])
debug_mode = st.sidebar.checkbox("🛠️ Enable Debug Mode")

# Main logic
if ticker:
    try:
        with st.spinner("Fetching data..."):
            df = fetch_stock_data(ticker)
            if df.empty or "Close" not in df.columns:
                raise ValueError("No valid data found for this ticker.")

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
        st.success(f"📊 Average Sentiment Score ({sentiment_model}): `{sentiment_score:.3f}`  —  Label: {label}")

        if debug_mode:
            st.markdown("#### 🧪 Debug: Sentiment Analysis Details")
            st.code("Raw Headlines:", language="text")
            st.write(headlines)
            st.write("🔍 Sentiment Scores:")
            st.write({
                "TextBlob": raw_sentiment,
                "VADER": fetch_sentiment_vader(headlines)[0],
                "BERT": fetch_sentiment_bert(headlines)
            })
            st.write("📋 Merged DataFrame Preview:")
            st.write(df.tail(10))

        st.subheader("🔮 Forecast")

        if compare_models:
            forecast_linear = train_linear_model(df)
            forecast_prophet = train_prophet_model(df)
            forecast_xgb = train_xgboost_model(df)

            merged = pd.merge(
                forecast_linear.rename(columns={"PredictedClose": "Linear"}),
                forecast_prophet[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Prophet"}),
                on="Date", how="inner"
            )
            merged = pd.merge(merged, forecast_xgb.rename(columns={"PredictedClose": "XGBoost"}), on="Date", how="inner")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Historical", color="black")
            ax.plot(merged["Date"], merged["Linear"], label="Linear Regression", linestyle="--", color="orange")
            ax.plot(merged["Date"], merged["Prophet"], label="Prophet", linestyle="--", color="green")
            ax.plot(merged["Date"], merged["XGBoost"], label="XGBoost", linestyle="--", color="blue")
            ax.set_title(f"{ticker.upper()} Forecast Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("### 📅 Forecast Comparison Table")
            st.dataframe(merged.tail(forecast_days))

            csv = merged.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Comparison CSV", data=csv, file_name=f"{ticker}_comparison_forecast.csv", mime="text/csv")

            if debug_mode:
                true = df["Close"].tail(forecast_days).values
                scores = {
                    "Linear Regression": (mean_absolute_percentage_error(true, merged["Linear"].head(forecast_days).values) * 100, np.sqrt(mean_squared_error(true, merged["Linear"].head(forecast_days).values))),
                    "Prophet": (mean_absolute_percentage_error(true, merged["Prophet"].head(forecast_days).values) * 100, np.sqrt(mean_squared_error(true, merged["Prophet"].head(forecast_days).values))),
                    "XGBoost": (mean_absolute_percentage_error(true, merged["XGBoost"].head(forecast_days).values) * 100, np.sqrt(mean_squared_error(true, merged["XGBoost"].head(forecast_days).values)))
                }

                st.markdown("#### 📊 Model Performance Comparison")
                for model, (mape, rmse) in scores.items():
                    st.write(f"**{model}**")
                    st.write(f"MAPE: {mape:.2f}%")
                    st.write(f"RMSE: {rmse:.2f}")

                best_model = min(scores.items(), key=lambda x: x[1][0])[0]
                best_score = scores[best_model][0]
                st.success(f"🏆 Best Model: {best_model} (MAPE: {best_score:.2f}%)")

                perf_fig = go.Figure()
                for metric_idx, metric_name in enumerate(["MAPE", "RMSE"]):
                    perf_fig.add_trace(go.Bar(
                        name=metric_name,
                        x=list(scores.keys()),
                        y=[v[metric_idx] for v in scores.values()]
                    ))
                perf_fig.update_layout(
                    title="Model Performance (MAPE and RMSE)",
                    barmode='group',
                    xaxis_title="Model",
                    yaxis_title="Score",
                    height=400
                )
                st.plotly_chart(perf_fig, use_container_width=True)

        else:
            if model_choice == "Linear Regression":
                forecast_df = train_linear_model(df)
                st.pyplot(plot_linear_forecast(df, forecast_df))
            elif model_choice == "Prophet":
                forecast_df = train_prophet_model(df)
                st.pyplot(plot_prophet_forecast(df, forecast_df))
                forecast_df = forecast_df.rename(columns={"ds": "Date", "yhat": "PredictedClose"})
            else:
                forecast_df = train_xgboost_model(df)
                st.pyplot(plot_xgboost_forecast(df, forecast_df))

            st.markdown(f"### 📅 Forecasted Prices ({model_choice})")
            st.dataframe(forecast_df.tail(forecast_days))

            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name=f"{ticker}_{model_choice.lower()}_forecast.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Oops! Something went wrong with the ticker '{ticker}'. Please make sure it's a valid stock symbol.")
        st.stop()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Built by Evang2")
