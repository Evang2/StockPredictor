import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import timedelta

def train_linear_model(df):
    # Copy and prepare data
    df = df.copy()
    df['DateOrd'] = df['Date'].map(pd.Timestamp.toordinal)
    df['Close_lag1'] = df['Close'].shift(1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Ensure features exist and drop missing rows
    df = df[['Date', 'DateOrd', 'Close', 'Close_lag1', 'MA7', 'DayOfWeek', 'VIX', 'Sentiment']].dropna()

    # Features and target
    features = ['DateOrd', 'Close_lag1', 'MA7', 'DayOfWeek', 'VIX', 'Sentiment']
    X = df[features]
    y = df['Close']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model with Ridge regression for better generalization
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Forecast next 30 days
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
        'Close_lag1': last_close,  # Assume constant lag
        'MA7': last_ma7,           # Assume constant MA
        'DayOfWeek': [d.weekday() for d in future_dates],
        'VIX': last_vix,
        'Sentiment': last_sentiment
    })

    # Predict
    X_future = future_df[features]
    X_future_scaled = scaler.transform(X_future)
    future_df['PredictedClose'] = model.predict(X_future_scaled)

    return future_df[['Date', 'PredictedClose']]
