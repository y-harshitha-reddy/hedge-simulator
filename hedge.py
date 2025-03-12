import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated asset data
data = {
    "Asset": ["AAPL", "TSLA", "GOOGL", "BTC", "ETH", "AMZN", "MSFT"],
    "Price": [175, 700, 2800, 45000, 3200, 3400, 300],
    "Volatility": [0.02, 0.05, 0.03, 0.08, 0.07, 0.025, 0.02]
}
assets = pd.DataFrame(data)

st.title("Hedge Fund Simulator")

# User input for portfolio allocation
st.sidebar.header("Allocate Funds")
investment = {}
total_funds = st.sidebar.number_input("Total Capital ($)", min_value=1000, value=100000, step=1000)

for asset in assets["Asset"]:
    investment[asset] = st.sidebar.number_input(f"Allocate to {asset} ($)", min_value=0, value=0, step=100)

total_allocated = sum(investment.values())
st.sidebar.write(f"**Total Allocated:** ${total_allocated}")

if total_allocated > total_funds:
    st.sidebar.error("Allocation exceeds available funds!")

# Simulating performance
days = 30
performance = {}
for asset in assets["Asset"]:
    price = assets.loc[assets["Asset"] == asset, "Price"].values[0]
    vol = assets.loc[assets["Asset"] == asset, "Volatility"].values[0]
    shares = investment[asset] / price if price else 0
    returns = np.cumsum(np.random.randn(days) * vol * price)
    performance[asset] = price + returns

df_performance = pd.DataFrame(performance)

# Plot performance
st.subheader("Portfolio Performance Over 30 Days")
fig, ax = plt.subplots()
df_performance.plot(ax=ax)
plt.xlabel("Days")
plt.ylabel("Asset Price")
st.pyplot(fig)

# Portfolio final value
final_values = {asset: investment[asset] * (df_performance[asset].iloc[-1] / assets.loc[assets["Asset"] == asset, "Price"].values[0]) for asset in assets["Asset"]}
total_final_value = sum(final_values.values())

st.subheader("Portfolio Summary")
st.write(f"Initial Capital: ${total_funds}")
st.write(f"Final Portfolio Value: ${total_final_value:.2f}")
st.write(f"Net Gain/Loss: ${total_final_value - total_funds:.2f}")

# Show asset-wise final value
st.subheader("Final Asset Allocation")
final_df = pd.DataFrame(final_values.items(), columns=["Asset", "Final Value ($)"])
st.dataframe(final_df)
