import yfinance as yf
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

ticker = "AAPL"
df = yf.Ticker(ticker).history(period="1y")
df.reset_index(inplace=True)
df.to_csv("data/aapl.csv", index=False)

print("✅ AAPL demo data saved to data/aapl.csv")
