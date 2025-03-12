import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# User input for investment details
st.sidebar.header("Investor Profile")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=18, value=30)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=1000, value=100000, step=1000)
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short-term", "Long-term"])
preferences = st.sidebar.multiselect("Investment Preferences", ["Equities", "Bonds", "Crypto", "Alternative Investments"], default=["Equities", "Bonds"])

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

st.title("Advanced Hedge Fund Simulator")

# Fetch real-time data
def fetch_market_data(tickers):
    data = yf.download(tickers, period="1mo", interval="1d")
    return data["Adj Close"]

market_data = fetch_market_data(list(assets.keys()))

# Portfolio Allocation Model
allocations = {}
for asset in assets.keys():
    allocations[asset] = st.sidebar.slider(f"Allocate % to {assets[asset]}", 0, 100, 10)

total_allocation = sum(allocations.values())
if total_allocation != 100:
    st.sidebar.warning("Total allocation should be 100%")

# Calculate Expected Returns & Risk Level
returns = market_data.pct_change().dropna()
expected_returns = returns.mean() * 252
risk = returns.std() * np.sqrt(252)

# Portfolio Performance
daily_portfolio_returns = returns.dot(np.array([allocations[a]/100 for a in assets.keys()]))
total_portfolio_return = (1 + daily_portfolio_returns).cumprod()

# Performance Metrics
sharpe_ratio = (daily_portfolio_returns.mean() / daily_portfolio_returns.std()) * np.sqrt(252)

# Visualization
st.subheader("Portfolio Performance Over 30 Days")
fig = px.line(total_portfolio_return, title="Portfolio Cumulative Returns")
st.plotly_chart(fig)

# Risk vs Return Scatter Plot
risk_return_df = pd.DataFrame({"Asset": list(assets.values()), "Expected Return": expected_returns.values, "Risk": risk.values})
fig_risk_return = px.scatter(risk_return_df, x="Risk", y="Expected Return", text="Asset", title="Risk vs Return Analysis")
st.plotly_chart(fig_risk_return)

# Sharpe Ratio
st.subheader("Sharpe Ratio")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Pie Chart for Portfolio Allocation
fig_pie = px.pie(names=list(assets.values()), values=list(allocations.values()), title="Portfolio Allocation Breakdown")
st.plotly_chart(fig_pie)

# Hedge Fund Strategy Simulator
st.subheader("Hedge Fund Strategy Simulator")
strategy = st.selectbox("Choose a Strategy", ["Long/Short Equity", "Arbitrage", "Event-Driven"])
st.write(f"AI-based strategy recommendation for {strategy} will be implemented here.")

# Market Trends & News Placeholder
st.subheader("Market Trends & News")
st.write("(Real-time market news integration can be added using APIs like Alpha Vantage or News API)")

st.write("---")
st.subheader("Portfolio Summary")
st.write(f"Investor: {name}, Age: {age}")
st.write(f"Risk Tolerance: {risk_tolerance}, Investment Horizon: {investment_horizon}")
st.write(f"Total Investment: ${investment_amount}")
