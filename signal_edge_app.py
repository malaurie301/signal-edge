import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# App title and placeholder logo
st.set_page_config(layout="centered")
st.title("SignalEdge — S&P 500 Signal App")

# Optional artwork (placeholder)
st.image("https://via.placeholder.com/728x90.png?text=SignalEdge+Logo", use_column_width=True)

# Load default data if no file uploaded
@st.cache_data
def load_data():
    df = pd.read_csv("sp500.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

# Calculate indicators and signals
def compute_indicators(df, sma_period):
    df["sma"] = df["close"].rolling(sma_period).mean()
    df["signal"] = 0
    df.loc[df["close"].shift(1) > df["sma"].shift(1), "signal"] = 1
    df.loc[df["close"].shift(1) < df["sma"].shift(1), "signal"] = -1
    df["position"] = df["signal"].shift(1)
    df["return"] = df["close"].pct_change()
    df["strategy_return"] = df["position"] * df["return"]
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
    return df

# Upload or load data
uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df.set_index("date", inplace=True)
else:
    df = load_data()

# SMA slider and calculation
sma_period = st.slider("Select SMA Period", min_value=5, max_value=200, value=50)
df = compute_indicators(df, sma_period)

# Strategy Overview
st.markdown("### Strategy Overview")
col1, col2 = st.columns(2)

with col1:
    total_return = df["cumulative_return"].iloc[-1] - 1 if "cumulative_return" in df.columns else 0
    st.metric("Total Strategy Return", f"{total_return:.2%}")

with col2:
    if "strategy_return" in df.columns and not df["strategy_return"].empty:
        sharpe = np.mean(df["strategy_return"]) / np.std(df["strategy_return"])
    else:
        sharpe = np.nan
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Chart with buy/sell arrows
st.markdown("### Price Chart with SMA and Buy/Sell Signals")
fig, ax = plt.subplots()
ax.plot(df.index, df["close"], label="Close Price", linewidth=1.5)
ax.plot(df.index, df["sma"], label=f"SMA {sma_period}", linestyle="--")

buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]
ax.scatter(buy_signals.index, buy_signals["close"], label="Buy Signal", color="green", marker="^", s=100)
ax.scatter(sell_signals.index, sell_signals["close"], label="Sell Signal", color="red", marker="v", s=100)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Signal Table
st.markdown("### Recent Buy/Sell Signals")
styled_df = df[["close", "sma", "signal", "position"]].tail(10)
st.dataframe(styled_df.style.highlight_max(axis=0))
