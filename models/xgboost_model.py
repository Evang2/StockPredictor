import pandas as pd
import numpy as np
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

def train_xgboost_model(df):
    df = df.copy()
    df['DateOrd'] = df['Date'].map(pd.Timestamp.toordinal)
    df['Close_lag1'] = df['Close'].shift(1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df = df[['Date', 'DateOrd', 'Close', 'Close_lag1', 'MA7', 'DayOfWeek', 'VIX', 'Sentiment']].dropna()

    features = ['DateOrd', 'Close_lag1', 'MA7', 'DayOfWeek', 'VIX', 'Sentiment']
    X = df[features]
    y = df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y)

    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_ord = [d.toordinal() for d in future_dates]

    last_close = df['Close'].iloc[-1]
    last_ma7 = df['MA7'].iloc[-1]
    last_vix = df['VIX'].iloc[-1]
    last_sentiment = df['Sentiment'].iloc[-1]

    future_df = pd.DataFrame({
        'Date': future_dates,
        'DateOrd': future_ord,
        'Close_lag1': last_close,
        'MA7': last_ma7,
        'DayOfWeek': [d.weekday() for d in future_dates],
        'VIX': last_vix,
        'Sentiment': last_sentiment
    })

    X_future = future_df[features]
    X_future_scaled = scaler.transform(X_future)
    future_df['PredictedClose'] = model.predict(X_future_scaled)

    return future_df[['Date', 'PredictedClose']]
