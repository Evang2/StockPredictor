import matplotlib.pyplot as plt


def plot_historical_prices(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Historical Price", color="blue")
    ax.set_title("Historical Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    return fig


def plot_linear_forecast(df_original, df_forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_original["Date"], df_original["Close"], label="Historical", color="blue")
    ax.plot(
        df_forecast["Date"],
        df_forecast["PredictedClose"],
        label="Forecast (Linear)",
        linestyle="--",
        color="orange",
    )
    ax.set_title("Linear Regression Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    return fig


def plot_prophet_forecast(df_original, forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_original["Date"], df_original["Close"], label="Historical", color="blue")
    ax.plot(
        forecast["ds"],
        forecast["yhat"],
        label="Forecast (Prophet)",
        linestyle="--",
        color="green",
    )
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="green",
        alpha=0.2,
        label="Confidence Interval",
    )
    ax.set_title("Prophet Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    return fig


def plot_with_moving_average(df, window=20):
    df = df.copy()
    df["MA"] = df["Close"].rolling(window=window).mean()
    df["DailyChange"] = df["Close"].pct_change() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Close"], label="Close Price")
    ax.plot(df["Date"], df["MA"], label=f"{window}-Day MA", linestyle="--")
    ax.set_title(f"Price + {window}-Day Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    return fig


def plot_xgboost_forecast(df_original, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_original["Date"], df_original["Close"], label="Historical", color="blue")
    ax.plot(
        forecast_df["Date"],
        forecast_df["PredictedClose"],
        label="Forecast (XGBoost)",
        linestyle="--",
        color="red",
    )
    ax.set_title("XGBoost Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)
    return fig
