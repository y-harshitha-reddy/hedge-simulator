import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated asset data
data = {
    "Asset": ["AAPL", "TSLA", "GOOGL", "BTC", "ETH", "AMZN", "MSFT"],
    "Price": [175, 700, 2800, 45000, 3200, 3400, 300],
    "Volatility": [0.02, 0.05, 0.03, 0.08, 0.07, 0.025, 0.02],
    "Expected Return": [0.07, 0.15, 0.12, 0.20, 0.18, 0.10, 0.08]
}
assets = pd.DataFrame(data)

st.title("Advanced Hedge Fund Simulator")

# User input for portfolio allocation
st.sidebar.header("Allocate Funds")
total_funds = st.sidebar.number_input("Total Capital ($)", min_value=1000, value=100000, step=1000)

investment = {}
for asset in assets["Asset"]:
    investment[asset] = st.sidebar.number_input(f"Allocate to {asset} ($)", min_value=0, value=0, step=100)

total_allocated = sum(investment.values())
st.sidebar.write(f"**Total Allocated:** ${total_allocated}")

if total_allocated > total_funds:
    st.sidebar.error("Allocation exceeds available funds!")

# Simulating performance
days = 30
performance = {}
final_values = {}
returns_list = []

for asset in assets["Asset"]:
    price = assets.loc[assets["Asset"] == asset, "Price"].values[0]
    vol = assets.loc[assets["Asset"] == asset, "Volatility"].values[0]
    expected_return = assets.loc[assets["Asset"] == asset, "Expected Return"].values[0]
    shares = investment[asset] / price if price else 0
    daily_returns = np.random.normal(loc=expected_return/252, scale=vol/np.sqrt(252), size=days)
    returns_list.append(daily_returns)
    price_series = price * (1 + np.cumsum(daily_returns))
    performance[asset] = price_series
    final_values[asset] = shares * price_series[-1]

df_performance = pd.DataFrame(performance)

# Plot performance
st.subheader("Portfolio Performance Over 30 Days")
fig, ax = plt.subplots()
df_performance.plot(ax=ax)
plt.xlabel("Days")
plt.ylabel("Asset Price")
st.pyplot(fig)

# Portfolio final value
total_final_value = sum(final_values.values())
net_gain = total_final_value - total_funds
st.subheader("Portfolio Summary")
st.write(f"Initial Capital: ${total_funds}")
st.write(f"Final Portfolio Value: ${total_final_value:.2f}")
st.write(f"Net Gain/Loss: ${net_gain:.2f} ({(net_gain/total_funds)*100:.2f}%)")

# Show asset-wise final value
st.subheader("Final Asset Allocation")
final_df = pd.DataFrame(final_values.items(), columns=["Asset", "Final Value ($)"])
st.dataframe(final_df)

# Risk and Return Analysis
st.subheader("Risk and Return Analysis")
avg_daily_returns = np.mean(np.array(returns_list), axis=1)
risk = np.std(np.array(returns_list), axis=1)
risk_return_df = pd.DataFrame({"Asset": assets["Asset"], "Average Daily Return": avg_daily_returns, "Risk (Std Dev)": risk})
st.dataframe(risk_return_df)

# Sharpe Ratio Calculation
risk_free_rate = 0.02 / 252
sharpe_ratios = (avg_daily_returns - risk_free_rate) / risk
sharpe_df = pd.DataFrame({"Asset": assets["Asset"], "Sharpe Ratio": sharpe_ratios})
st.subheader("Sharpe Ratios (Higher is Better)")
st.dataframe(sharpe_df)
