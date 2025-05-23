import os
from dotenv import load_dotenv
import requests
import yfinance as yf
import pandas as pd
import wbdata
from datetime import datetime
import streamlit as st

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")


def fetch_stock_data(ticker, period="1y", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df.reset_index(inplace=True)

        if df.empty:
            raise ValueError("Empty DataFrame returned")

        return df

    except Exception as e:
        # Fallback to local demo file
        fallback_path = f"data/{ticker.lower()}.csv"
        if os.path.exists(fallback_path):
            try:
                df = pd.read_csv(fallback_path, parse_dates=["Date"])
                return df
            except Exception as err:
                print(f"❌ Error loading fallback CSV for {ticker}: {err}")
                return pd.DataFrame()
        else:
            print(f"❌ Ticker {ticker} not found and no fallback CSV available.")
            return pd.DataFrame()


def fetch_vix_data(period="1y"):
    vix = yf.Ticker("^VIX")
    df = vix.history(period=period)
    df.reset_index(inplace=True)
    df = df.rename(columns={"Close": "VIX"})
    return df[["Date", "VIX"]]


def fetch_fred_series(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    res = requests.get(url).json()
    data = res.get("observations", [])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].rename(columns={"date": "Date", "value": series_id})


def fetch_macro_indicators():
    indicators = ["FEDFUNDS", "CPIAUCSL", "UNRATE"]
    dfs = [fetch_fred_series(ind) for ind in indicators]
    macro = dfs[0]
    for df in dfs[1:]:
        macro = macro.merge(df, on="Date", how="outer")
    return macro.fillna(method="ffill")


def fetch_gdp_worldbank(country_code="USA"):
    df = wbdata.get_dataframe(
        {"NY.GDP.MKTP.CD": "GDP"},
        country=country_code,
        data_date=(datetime(2010, 1, 1), datetime.today()),
    ).reset_index()
    df = df.rename(columns={"date": "Date"})
    df = df.sort_values("Date")
    return df
