import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Simulated asset data
data = {
    "Asset": ["AAPL", "TSLA", "GOOGL", "BTC-USD", "ETH-USD", "AMZN", "MSFT"],
    "Sector": ["Tech", "Auto", "Tech", "Crypto", "Crypto", "Retail", "Tech"]
}
assets = pd.DataFrame(data)

st.title("Hedge Fund Simulator - One Stop Finance Platform")

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

# Fetch real-time price data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

def fetch_data(asset):
    try:
        df = yf.download(asset, start=start_date, end=end_date)["Adj Close"]
        return df
    except:
        return np.full(30, np.nan)

st.sidebar.subheader("Fetching Real-Time Data...")
performance = {asset: fetch_data(asset) for asset in assets["Asset"]}
df_performance = pd.DataFrame(performance)

# Plot performance
st.subheader("Portfolio Performance Over Last 30 Days")
fig, ax = plt.subplots()
df_performance.plot(ax=ax)
plt.xlabel("Days")
plt.ylabel("Asset Price")
st.pyplot(fig)

# Portfolio final value
final_values = {asset: investment[asset] * (df_performance[asset].iloc[-1] / df_performance[asset].iloc[0]) for asset in assets["Asset"]}
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
risk_measures = {asset: df_performance[asset].pct_change().std() * np.sqrt(252) for asset in assets["Asset"]}
st.write(pd.DataFrame(risk_measures.items(), columns=["Asset", "Annualized Volatility"]))

# Sharpe ratio calculation (assuming risk-free rate = 2%)
st.subheader("Sharpe Ratio")
risk_free_rate = 0.02 / 252
sharpe_ratios = {}
for asset in assets["Asset"]:
    mean_return = df_performance[asset].pct_change().mean()
    std_dev = df_performance[asset].pct_change().std()
    sharpe_ratios[asset] = (mean_return - risk_free_rate) / std_dev if std_dev else 0
st.write(pd.DataFrame(sharpe_ratios.items(), columns=["Asset", "Sharpe Ratio"]))

# Live news feed
st.subheader("Latest Financial News")
st.write("Fetching latest market updates...")
st.write("(Feature to be expanded with live API integration)")

# Economic indicators
st.subheader("Economic Indicators")
st.write("(Future development: Integrating GDP growth, inflation, interest rates, etc.)")

# Conclusion and Insights
st.subheader("Final Analysis & Insights")
if total_final_value > total_funds:
    st.success("Your portfolio has gained value! Consider reinvesting profits or diversifying further.")
elif total_final_value < total_funds:
    st.warning("Your portfolio has lost value. Evaluate risk exposure and consider safer assets.")
else:
    st.info("Your portfolio remained stable. Consider adjusting your strategy for better returns.")

st.write("**Key Takeaways:**")
st.write("- The portfolio's overall performance depends on asset allocation and market conditions.")
st.write("- Higher Sharpe ratios indicate better risk-adjusted returns.")
st.write("- Diversifying across sectors reduces overall risk exposure.")
st.write("- Monitoring economic indicators can improve investment decisions.")
