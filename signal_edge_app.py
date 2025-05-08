

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("SignalEdge — Smarter Market Timing")

st.sidebar.header("Strategy Settings")
sma_period = st.sidebar.slider("SMA Period", 10, 200, 50, step=5)
min_change = st.sidebar.slider("Min % Price Change (Signal Trigger)", 0.5, 5.0, 1.5)
min_days_between_signals = st.sidebar.slider("Min Days Between Signals", 5, 30, 10)
cash_yield = st.sidebar.selectbox("Cash Yield When Out (Annual %)", [1.0, 2.5, 4.0], index=1)

@st.cache_data
def load_data():
    spy = yf.download("SPY", start="2010-01-01")
    vix = yf.download("^VIX", start="2010-01-01")
    df = spy[["Close"]].rename(columns={"Close": "close"})
    df["vix"] = vix["Close"]
    df.dropna(inplace=True)
    return df

df = load_data()
df["sma"] = df["close"].rolling(sma_period).mean()
df["pct_change"] = df["close"].pct_change(periods=min_days_between_signals).shift(-min_days_between_signals)
df["signal"] = 0

last_signal_day = -min_days_between_signals
for i in range(1, len(df)):
    if i - last_signal_day < min_days_between_signals:
        continue
    upward = df["close"].iloc[i] > df["sma"].iloc[i]
    down_from_below = df["close"].iloc[i-1] <= df["sma"].iloc[i-1]
    if upward and down_from_below and df["pct_change"].iloc[i] > min_change / 100:
        df.loc[df.index[i], "signal"] = 1
        last_signal_day = i
    elif df["close"].iloc[i] < df["sma"].iloc[i] and df["close"].iloc[i-1] >= df["sma"].iloc[i-1]:
        df.loc[df.index[i], "signal"] = -1
        last_signal_day = i

df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)
df["daily_return"] = df["close"].pct_change()
daily_cash_rate = (1 + cash_yield / 100) ** (1/252) - 1
df["strategy_return"] = np.where(df["position"].shift(1) == 1, df["daily_return"], daily_cash_rate)
df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
df["cumulative_market"] = (1 + df["daily_return"].fillna(0)).cumprod()

total_return = df["cumulative_strategy"].iloc[-1] - 1
market_return = df["cumulative_market"].iloc[-1] - 1
drawdown = (df["cumulative_strategy"].cummax() - df["cumulative_strategy"]).max()
sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * np.sqrt(252)

st.subheader("Performance Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Strategy Return", f"{total_return:.2%}")
col2.metric("Max Drawdown", f"{drawdown:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")

st.subheader("Signal Chart with SMA")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df["close"], label="Close Price", alpha=0.7)
ax.plot(df.index, df["sma"], label=f"SMA {sma_period}", linestyle="--")
ax.scatter(df[df["signal"] == 1].index, df[df["signal"] == 1]["close"], marker="^", color="green", label="Buy Signal")
ax.scatter(df[df["signal"] == -1].index, df[df["signal"] == -1]["close"], marker="v", color="red", label="Sell Signal")
ax.legend()
st.pyplot(fig)

st.subheader("Cumulative Return: Strategy vs. Market")
fig2, ax2 = plt.subplots()
ax2.plot(df.index, df["cumulative_strategy"], label="SignalEdge Strategy")
ax2.plot(df.index, df["cumulative_market"], label="S&P 500 Buy & Hold", linestyle="--")
ax2.legend()
st.pyplot(fig2)

st.subheader("Recent Signals")
st.dataframe(df[["close", "sma", "vix", "signal", "position"]].tail(25).reset_index())
