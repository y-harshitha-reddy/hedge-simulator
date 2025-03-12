import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated asset data
data = {
    "Asset": ["AAPL", "TSLA", "GOOGL", "BTC", "ETH", "AMZN", "MSFT"],
    "Price": [175, 700, 2800, 45000, 3200, 3400, 300],
    "Volatility": [0.02, 0.05, 0.03, 0.08, 0.07, 0.025, 0.02],
    "Sector": ["Tech", "Auto", "Tech", "Crypto", "Crypto", "Retail", "Tech"]
}
assets = pd.DataFrame(data)

st.title("Hedge Fund Simulator")

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
np.random.seed(42)  # For reproducibility
for asset in assets["Asset"]:
    price = assets.loc[assets["Asset"] == asset, "Price"].values[0]
    vol = assets.loc[assets["Asset"] == asset, "Volatility"].values[0]
    shares = investment[asset] / price if price else 0
    returns = np.cumsum(np.random.normal(0, vol * price, days))
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

# Sector-wise allocation
st.subheader("Sector-Wise Allocation")
sector_allocation = assets.set_index("Asset")["Sector"].to_dict()
final_df["Sector"] = final_df["Asset"].map(sector_allocation)
sector_summary = final_df.groupby("Sector")["Final Value ($)"].sum()
st.bar_chart(sector_summary)

# Risk analysis
st.subheader("Risk Analysis")
risk_measures = {}
for asset in assets["Asset"]:
    risk_measures[asset] = df_performance[asset].pct_change().std() * np.sqrt(252)
st.write(pd.DataFrame(risk_measures.items(), columns=["Asset", "Annualized Volatility"]))

# Sharpe ratio calculation (assuming risk-free rate = 2%)
st.subheader("Sharpe Ratio")
risk_free_rate = 0.02 / 252  # Daily risk-free rate
sharpe_ratios = {}
for asset in assets["Asset"]:
    mean_return = df_performance[asset].pct_change().mean()
    std_dev = df_performance[asset].pct_change().std()
    sharpe_ratios[asset] = (mean_return - risk_free_rate) / std_dev if std_dev else 0
st.write(pd.DataFrame(sharpe_ratios.items(), columns=["Asset", "Sharpe Ratio"]))
