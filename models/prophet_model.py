from prophet import Prophet
import pandas as pd


def train_prophet_model(df):
    df = df.copy()
    df = df[["Date", "Close", "VIX", "Sentiment"]].dropna()
    df = df.rename(columns={"Date": "ds", "Close": "y"})

    # Initialize Prophet with tuned parameters
    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
    )

    # Add custom regressors
    model.add_regressor("VIX")
    model.add_regressor("Sentiment")

    # Fit model
    model.fit(df)

    # Make future dataframe
    future = model.make_future_dataframe(periods=30)
    future["VIX"] = df["VIX"].iloc[-1]  # Assume last known VIX value
    future["Sentiment"] = df["Sentiment"].iloc[-1]  # Assume last known sentiment

    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
