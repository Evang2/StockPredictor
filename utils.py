import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np


def label_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def add_sentiment_column(df, score):
    df = df.copy()
    df["SentimentLabel"] = label_sentiment(score)
    return df


def merge_all_features(stock_df, vix_df, macro_df, sentiment_score):
    df = stock_df.copy()

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.tz_localize(None)
    macro_df["Date"] = pd.to_datetime(macro_df["Date"]).dt.tz_localize(None)

    df = df.merge(vix_df, on="Date", how="left")
    df = df.merge(macro_df, on="Date", how="left")
    df["Sentiment"] = sentiment_score
    df.fillna(method="ffill", inplace=True)
    return df


def evaluate_model(true, predicted):
    mape = mean_absolute_percentage_error(true, predicted) * 100
    rmse = np.sqrt(mean_squared_error(true, predicted))
    return round(mape, 2), round(rmse, 2)
