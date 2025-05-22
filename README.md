# Smart Stock Predictor 📈

A powerful Streamlit web application that forecasts stock prices using machine learning models combined with live news sentiment, macroeconomic indicators (like VIX), and financial data.

---

## 🚀 Features

- 🔎 Input any stock ticker (e.g., `AAPL`, `MSFT`, `TSLA`)
- 🧠 Models used:
  - Linear Regression
  - Facebook Prophet (with external regressors)
  - XGBoost
- 📰 Sentiment analysis from financial headlines:
  - TextBlob
  - VADER
  - DistilBERT (HuggingFace)
- 📊 visualizations (Plotly) 
- 🧪 Debug mode with:
  - Raw sentiment scores
  - Model MAPE & RMSE
  - Forecast table & download buttons
- 🏆 Best Model badge based on performance

---

## 📂 Project Structure

```
StockPredictor/
├── app.py                  # Main Streamlit app
├── data_fetcher.py         # Fetch stock, VIX, macroeconomic data
├── sentiment_analyzer.py   # Sentiment analysis (news scraping + NLP)
├── utils.py                # Merging, preprocessing, etc
├── visualizer.py           # All charts and Plotly graphs
├── models/
│   ├── linear_model.py     # Linear Regression trainer
│   ├── prophet_model.py    # Prophet model trainer
│   └── xgboost_model.py    # XGBoost model trainer
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it
```
---
## 🖼️ Screenshots

![Screenshot 2025-05-23 021729](https://github.com/user-attachments/assets/2b34d3cc-c1c2-4f46-b795-3f8b5d95c27e)
![Screenshot 2025-05-23 021740](https://github.com/user-attachments/assets/d79eeeed-d14d-4c61-a8eb-b91a0e108a35)
![Screenshot 2025-05-23 021748](https://github.com/user-attachments/assets/f0db939a-3d82-469d-8472-18dacf548189)
![Screenshot 2025-05-23 021800](https://github.com/user-attachments/assets/2ad6d927-c807-4310-bb55-de7179c14bf3)
![Screenshot 2025-05-23 021816](https://github.com/user-attachments/assets/2eee19fc-2c84-430f-8cd8-dd2b7f76ffd3)
![Screenshot 2025-05-23 021833](https://github.com/user-attachments/assets/e109cd62-f139-432f-af6b-d6b98e6613c3)
![Screenshot 2025-05-23 021910](https://github.com/user-attachments/assets/10f365b3-3492-410f-a35c-3c3693279a0a)
![Screenshot 2025-05-23 021940](https://github.com/user-attachments/assets/8af5e6f4-10c0-4818-a22b-c0c101e30234)
![Screenshot 2025-05-23 022002](https://github.com/user-attachments/assets/183e0c68-70dd-4883-adde-6e8b586b4d56)

---

## 📦 Installation

```bash
# 1. Clone the repo
https://github.com/YOUR_USERNAME/StockPredictor.git

cd StockPredictor

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🌍 APIs & Integrations

- [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/) – historical data
- [Finnhub.io](https://finnhub.io/) – financial news headlines (free tier)
- [FRED API](https://fred.stlouisfed.org/) – macroeconomic indicators (interest rate, CPI, etc)

To use FRED or Finnhub:
1. Create a `.env` file:
   ```env
   FRED_API_KEY=your_key_here
   FINNHUB_API_KEY=your_key_here
   ```
2. Load with `python-dotenv` or `os.environ.get()`

---

## 📊 Model Evaluation (Debug Mode)

| Model            | MAPE ↓   | RMSE ↓   |
|------------------|----------|----------|
| Linear Regression| ~2-5%     | Moderate |
| Prophet          | ~3-4%     | Good     |
| XGBoost          | **Best**  | Excellent|

> Choose “Compare All Models” + Enable Debug Mode for detailed metrics.

---

## 📥 Export & Forecast Tables

Every forecast can be downloaded as CSV using `st.download_button()`.
Forecasts include:
- Date
- Predicted Close
- (Prophet only) Confidence bounds


---

## 🤝 Contributing

Feel free to fork the repo and open PRs!

1. Clone
2. Create a feature branch
3. Push your feature + make PR

---

## 📄 License

MIT License. See `LICENSE` file.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is **not suitable for commercial use or real-world trading decisions**. Forecasts are based on public data and machine learning models which may have limitations.

---

## ✨ Credits

- Built by Evang2
- Icons: Streamlit + Plotly
- NLP: HuggingFace Transformers
- Data APIs: Yahoo Finance, FRED, Finnhub
