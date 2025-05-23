import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import streamlit as st
import os
from dotenv import load_dotenv

try:
    FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
except Exception:
    load_dotenv()
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# -------------------------
# 🔍 1. News Scraping (Google News) — for legacy fallback/testing
# -------------------------


def fetch_news_sentiment(ticker):
    try:
        url = (
            f"https://news.google.com/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        print("✅ Google News Status Code:", response.status_code)

        soup = BeautifulSoup(response.text, "html.parser")

        headlines = [a.text.strip() for a in soup.select("article h3 a")]

        if len(headlines) < 5:
            headlines += [h.text.strip() for h in soup.find_all("h3")]

        if len(headlines) < 5:
            headlines += [
                div.text.strip() for div in soup.find_all("div", {"role": "heading"})
            ]

        headlines = list(dict.fromkeys(headlines))[:10]

        if not headlines:
            print("❗ No headlines found for", ticker)
            return 0.0, []

        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_score = sum(scores) / len(scores)

        print("🔍 Headlines:", headlines)
        print("📊 TextBlob Scores:", scores)
        print("✅ Avg Sentiment:", avg_score)

        return avg_score, headlines

    except Exception as e:
        print("❌ Error in fetch_news_sentiment:", e)
        return 0.0, []


# -------------------------
# 🧠 2. VADER Sentiment
# -------------------------


def fetch_sentiment_vader(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg_score = sum(scores) / len(scores) if scores else 0
    label = (
        "Positive"
        if avg_score > 0.05
        else "Negative"
        if avg_score < -0.05
        else "Neutral"
    )
    return avg_score, label


# -------------------------
# 🤖 3. BERT Sentiment (HuggingFace distilBERT)
# -------------------------

bert_sentiment = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def fetch_sentiment_bert(headlines):
    results = bert_sentiment(headlines)
    scores = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score


# -------------------------
# 📰 4. Finnhub News + Sentiment (Recommended)
# -------------------------


def fetch_news_sentiment_finnhub(ticker):
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-12-01&to=2025-12-31&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        data = response.json()

        headlines = [item["headline"] for item in data if "headline" in item]
        headlines = list(dict.fromkeys(headlines))[:10]

        if not headlines:
            print(f"❗ No Finnhub headlines for {ticker}")
            return 0.0, []

        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_score = sum(scores) / len(scores)

        print("✅ Finnhub headlines:", headlines)
        print("📊 TextBlob scores:", scores)
        print("📈 Avg Sentiment (TextBlob):", avg_score)

        return avg_score, headlines

    except Exception as e:
        print("❌ Finnhub API error:", e)
        return 0.0, []
