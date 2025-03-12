import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    try:
        data = yf.download(tickers, period="1mo", interval="1d")
        if "Adj Close" in data.columns:
            return data["Adj Close"]
        else:
            st.warning("Adjusted Close prices not available. Using Close prices instead.")
            return data["Close"] if "Close" in data.columns else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

market_data = fetch_market_data(list(assets.keys()))

# Portfolio Allocation Model
allocations = {}
for asset in assets.keys():
    allocations[asset] = st.sidebar.slider(f"Allocate % to {assets[asset]}", 0, 100, 10)

total_allocation = sum(allocations.values())
if total_allocation != 100:
    st.sidebar.warning("Total allocation should be 100%")

# Calculate Expected Returns & Risk Level
if not market_data.empty:
    returns = market_data.pct_change().dropna()
    expected_returns = returns.mean() * 252
    risk = returns.std() * np.sqrt(252)

    # Portfolio Performance
    daily_portfolio_returns = returns.dot(np.array([allocations[a]/100 for a in assets.keys()]))
    total_portfolio_return = (1 + daily_portfolio_returns).cumprod()

    # Performance Metrics
    sharpe_ratio = (daily_portfolio_returns.mean() / daily_portfolio_returns.std()) * np.sqrt(252)

    # R-Square Calculation
    X = np.arange(len(daily_portfolio_returns)).reshape(-1, 1)
    y = daily_portfolio_returns.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r_square = r2_score(y, y_pred)

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

    # R-Square Value
    st.subheader("Model Accuracy (R-Square)")
    st.write(f"R-Square Value: {r_square:.4f}")

    # Pie Chart for Portfolio Allocation
    fig_pie = px.pie(names=list(assets.values()), values=list(allocations.values()), title="Portfolio Allocation Breakdown")
    st.plotly_chart(fig_pie)

    # AI-Based Long/Short Equity Strategy
    st.subheader("AI-Based Long/Short Equity Strategy")
    momentum = returns.mean()
    long_positions = momentum[momentum > 0].index.tolist()
    short_positions = momentum[momentum < 0].index.tolist()

    st.write("### Suggested Positions")
    st.write(f"**Long (Buy):** {', '.join(long_positions) if long_positions else 'None'}")
    st.write(f"**Short (Sell):** {', '.join(short_positions) if short_positions else 'None'}")

# Hedge Fund Strategy Simulator
st.subheader("Hedge Fund Strategy Simulator")
strategy = st.selectbox("Choose a Strategy", ["Long/Short Equity", "Arbitrage", "Event-Driven"])

st.write("---")
st.subheader("Portfolio Summary")
st.write(f"Investor: {name}, Age: {age}")
st.write(f"Risk Tolerance: {risk_tolerance}, Investment Horizon: {investment_horizon}")
st.write(f"Total Investment: ${investment_amount}")
