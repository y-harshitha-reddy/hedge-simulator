import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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

time_series = np.arange(1, days + 1)

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

df_performance = pd.DataFrame(performance, index=time_series)

# Live interactive line chart
st.subheader("Portfolio Performance Over 30 Days")
fig = px.line(df_performance, x=df_performance.index, y=df_performance.columns, title="Asset Price Movement Over Time")
fig.update_xaxes(title="Days")
fig.update_yaxes(title="Price ($)")
st.plotly_chart(fig)

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

# Pie Chart for Asset Allocation
fig_pie = px.pie(final_df, names='Asset', values='Final Value ($)', title="Portfolio Allocation Breakdown")
st.plotly_chart(fig_pie)

# Risk and Return Analysis
st.subheader("Risk and Return Analysis")
avg_daily_returns = np.mean(np.array(returns_list), axis=1)
risk = np.std(np.array(returns_list), axis=1)
risk_return_df = pd.DataFrame({"Asset": assets["Asset"], "Average Daily Return": avg_daily_returns, "Risk (Std Dev)": risk})
st.dataframe(risk_return_df)

# Interactive scatter plot for risk vs return
fig_risk_return = px.scatter(risk_return_df, x="Risk (Std Dev)", y="Average Daily Return", text="Asset", title="Risk vs Return Analysis", size_max=60)
fig_risk_return.update_traces(textposition='top center')
st.plotly_chart(fig_risk_return)

# Sharpe Ratio Calculation
risk_free_rate = 0.02 / 252
sharpe_ratios = (avg_daily_returns - risk_free_rate) / risk
sharpe_df = pd.DataFrame({"Asset": assets["Asset"], "Sharpe Ratio": sharpe_ratios})
st.subheader("Sharpe Ratios (Higher is Better)")
st.dataframe(sharpe_df)

# Interactive bar chart for Sharpe Ratios
fig_sharpe = px.bar(sharpe_df, x="Asset", y="Sharpe Ratio", title="Sharpe Ratios per Asset", color="Sharpe Ratio")
st.plotly_chart(fig_sharpe)

# Monte Carlo Simulation for Future Predictions
st.subheader("Monte Carlo Simulation for Future Prices")
simulations = 1000
future_days = 30
monte_carlo_results = {}

for asset in assets["Asset"]:
    last_price = assets.loc[assets["Asset"] == asset, "Price"].values[0]
    expected_return = assets.loc[assets["Asset"] == asset, "Expected Return"].values[0]
    volatility = assets.loc[assets["Asset"] == asset, "Volatility"].values[0]
    simulations_array = np.zeros((simulations, future_days))
    for sim in range(simulations):
        daily_returns = np.random.normal(expected_return/252, volatility/np.sqrt(252), future_days)
        price_series = last_price * (1 + np.cumsum(daily_returns))
        simulations_array[sim] = price_series
    monte_carlo_results[asset] = np.mean(simulations_array, axis=0)

df_monte_carlo = pd.DataFrame(monte_carlo_results, index=np.arange(1, future_days + 1))
fig_mc = px.line(df_monte_carlo, x=df_monte_carlo.index, y=df_monte_carlo.columns, title="Monte Carlo Simulated Future Prices")
fig_mc.update_xaxes(title="Days")
fig_mc.update_yaxes(title="Predicted Price ($)")
st.plotly_chart(fig_mc)
