

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("SignalEdge — S&P 500 Signal App")

# Sidebar — parameters
st.sidebar.header("Settings")
sma_period = st.sidebar.slider("Select SMA Period", 5, 200, 50)
cash_yield = st.sidebar.selectbox("Cash Yield (when out)", [1.0, 2.5, 4.0], index=1)

# Download SPY and VIX data
@st.cache_data
def load_data():
    spy = yf.download("SPY", start="2005-01-01")
    vix = yf.download("^VIX", start="2005-01-01")
    df = spy[["Close"]].rename(columns={"Close": "close"})
    df["vix"] = vix["Close"]
    df.dropna(inplace=True)
    return df

df = load_data()
df["sma"] = df["close"].rolling(sma_period).mean()

# Signal logic
df["signal"] = 0
df["position"] = 0
for i in range(1, len(df)):
    if df["close"].iloc[i] > df["sma"].iloc[i] and df["vix"].iloc[i] < 25:
        df.at[df.index[i], "signal"] = 1
    elif df["close"].iloc[i] < df["sma"].iloc[i] and df["vix"].iloc[i] > 25:
        df.at[df.index[i], "signal"] = -1

# Position tracking
position = 0
for i in range(len(df)):
    if df["signal"].iloc[i] == 1:
        position = 1
    elif df["signal"].iloc[i] == -1:
        position = 0
    df.at[df.index[i], "position"] = position

# Strategy performance
df["daily_return"] = df["close"].pct_change()
df["strategy_return"] = df["daily_return"] * df["position"]
df["cash_return"] = ((1 + (cash_yield / 100) / 252) ** (1 * (1 - df["position"])) - 1)
df["combined_return"] = df["strategy_return"] + df["cash_return"]

cumulative_return = (df["combined_return"] + 1).cumprod()
total_return = cumulative_return.iloc[-1] - 1
sharpe_ratio = df["combined_return"].mean() / df["combined_return"].std() * np.sqrt(252)

# Chart and metrics
st.subheader("Strategy Overview")
col1, col2 = st.columns(2)
col1.metric("Total Strategy Return", f"{total_return*100:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")

# Plot
fig, ax = plt.subplots()
ax.plot(df.index, df["close"], label="Close Price")
ax.plot(df.index, df["sma"], label=f"SMA {sma_period}", linestyle="--")
buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]
ax.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", label="Buy Signal")
ax.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", label="Sell Signal")
ax.legend()
st.pyplot(fig)

# Signal Table
st.subheader("Recent Buy/Sell Signals")
st.dataframe(df[["close", "sma", "vix", "signal", "position"]].tail(20).reset_index())

# Backtest chart
st.subheader("Cumulative Return")
fig2, ax2 = plt.subplots()
ax2.plot(cumulative_return, label="Strategy")
ax2.set_ylabel("Growth of $1")
ax2.legend()
st.pyplot(fig2)

# Footer
st.caption("SignalEdge — Trend + Volatility S&P 500 Timing Strategy")
