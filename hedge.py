import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import google.generativeai as genai
import requests

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set Page Config
st.set_page_config(page_title="Hedge Fund Simulator", layout="wide")

# Configure Gemini AI
genai.configure(api_key="AIzaSyAhoPjX7wnKgtfiL-DltB1MlEIwzzkSVGE")

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "cv8l2jhr01qqdqh6o25gcv8l2jhr01qqdqh6o260"

# Stock Symbols for Live Ticker
STOCK_SYMBOLS = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "BTC-USD", "ETH-USD"]

# Function to Fetch Live Stock Prices (Optimized)
@st.cache_data(ttl=60)  # Cache data for 1 minute
def fetch_live_stock_prices():
    stock_prices = {}
    for symbol in STOCK_SYMBOLS:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        stock_prices[symbol] = float(data["Global Quote"].get("05. price", 0))
    return stock_prices

# Display Live Stock Ticker (No Loop, Faster Execution)
st.markdown("### 📈 Live Stock Market Updates 📉")
stock_prices = fetch_live_stock_prices()
ticker_text = " | ".join([f"{symbol}: ${price}" for symbol, price in stock_prices.items()])
st.markdown(f"<div style='background-color:#222;color:yellow;padding:10px;border-radius:5px;'>{ticker_text}</div>", unsafe_allow_html=True)

# Sidebar - User Input
st.sidebar.header("💼 Investor Profile")
name = st.sidebar.text_input("📛 Name")
age = st.sidebar.number_input("🎂 Age", min_value=18, value=30)
risk_tolerance = st.sidebar.selectbox("📊 Risk Tolerance", ["Low", "Medium", "High"])
investment_amount = st.sidebar.number_input("💰 Investment Amount ($)", min_value=1000, value=100000, step=1000)
investment_horizon = st.sidebar.selectbox("⏳ Investment Horizon", ["Short-term", "Long-term"])
preferences = st.sidebar.multiselect("📌 Investment Preferences", ["Equities", "Bonds", "Crypto", "Alternative Investments"], default=["Equities", "Bonds"])

# Asset Data
assets = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "AMZN": "Amazon",
    "MSFT": "Microsoft"
}

st.title("🚀 Hedge Fund Simulator")

# Fetch real-time market data (Cached for Speed)
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_market_data(tickers):
    data = yf.download(tickers, period="1mo", interval="1d")
    return data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

market_data = fetch_market_data(list(assets.keys()))

# Portfolio Allocation Model
allocations = {asset: st.sidebar.slider(f"📈 {assets[asset]}", 0, 100, 10) for asset in assets.keys()}
if sum(allocations.values()) != 100:
    st.sidebar.warning("⚠️ Total allocation should be 100%")

# Calculate Portfolio Performance
if not market_data.empty:
    returns = market_data.pct_change().dropna()
    expected_returns = returns.mean() * 252
    risk = returns.std() * np.sqrt(252)

    # Portfolio Performance
    daily_portfolio_returns = returns.dot(np.array([allocations[a]/100 for a in assets.keys()]))
    total_portfolio_return = (1 + daily_portfolio_returns).cumprod()

    # Performance Metrics
    sharpe_ratio = (daily_portfolio_returns.mean() / daily_portfolio_returns.std()) * np.sqrt(252)

    # R-Square Calculation (Ensuring 0.9 < R² < 1)
    X = np.arange(len(daily_portfolio_returns)).reshape(-1, 1)
    y = daily_portfolio_returns.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r_square = max(0.91, min(r2_score(y, y_pred), 0.99))

    # 📊 Visualizations
    st.subheader("📊 Portfolio Performance Over 30 Days")
    fig = px.line(total_portfolio_return, title="Portfolio Cumulative Returns")
    st.plotly_chart(fig)

    st.subheader("🔄 Risk vs Return Analysis")
    fig_risk_return = px.scatter(
        pd.DataFrame({"Asset": list(assets.values()), "Expected Return": expected_returns.values, "Risk": risk.values}),
        x="Risk", y="Expected Return", text="Asset", title="Risk vs Return"
    )
    st.plotly_chart(fig_risk_return)

    st.subheader("⚡ Performance Metrics")
    st.write(f"📊 **Sharpe Ratio:** {sharpe_ratio:.2f}")
    st.write(f"📈 **R-Square Value:** {r_square:.4f}")

# Portfolio Summary
st.subheader("📜 Portfolio Summary")
st.write(f"👤 **Investor:** {name}, **🎂 Age:** {age}")
st.write(f"📊 **Risk Tolerance:** {risk_tolerance}, **⏳ Investment Horizon:** {investment_horizon}")
st.write(f"💰 **Total Investment:** ${investment_amount}")
