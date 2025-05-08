import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")
st.title("SignalEdge — S&P 500 Signal App")

# Sidebar Inputs
st.sidebar.header("Upload your price data (CSV with 'date' and 'close')")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
default_symbol = "^GSPC"

sma_period = st.sidebar.slider("Select SMA Period", 5, 200, 30)
cash_yield = st.sidebar.selectbox("Cash Yield (Annual)", [1.0, 2.5, 4.0, 5.0], index=1)

def load_data():
    df = yf.download(default_symbol, start="2022-01-01", end=datetime.today())
    df = df[['Close']].reset_index()
    df.columns = ['date', 'close']
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
else:
    df = load_data()

df['sma'] = df['close'].rolling(window=sma_period).mean()

df['signal'] = 0
df['position'] = 0

for i in range(1, len(df)):
    if df['close'][i] > df['sma'][i] and df['close'][i-1] <= df['sma'][i-1]:
        df.loc[i, 'signal'] = 1  # Buy
    elif df['close'][i] < df['sma'][i] and df['close'][i-1] >= df['sma'][i-1]:
        df.loc[i, 'signal'] = -1  # Sell

df['position'] = df['signal'].replace(0, method='ffill').fillna(0)

# Strategy return calculation
df['market_return'] = df['close'].pct_change()
df['strategy_return'] = df['market_return'] * df['position'].shift(1).fillna(0)
daily_cash_return = (1 + cash_yield / 100) ** (1/252) - 1
df['cash_return'] = np.where(df['position'].shift(1) == 0, daily_cash_return, 0)
df['combined_return'] = df['strategy_return'] + df['cash_return']
df['equity_curve'] = (1 + df['combined_return']).cumprod()

# Metrics
total_return = (df['equity_curve'].iloc[-1] - 1) * 100
sharpe_ratio = df['combined_return'].mean() / df['combined_return'].std() * np.sqrt(252)

# Chart
fig, ax = plt.subplots()
ax.plot(df['date'], df['close'], label="Close Price", linewidth=1)
ax.plot(df['date'], df['sma'], label=f"SMA {sma_period}", linestyle="--")
ax.scatter(df.loc[df['signal'] == 1, 'date'], df.loc[df['signal'] == 1, 'close'], label='Buy Signal', marker='^', color='green')
ax.scatter(df.loc[df['signal'] == -1, 'date'], df.loc[df['signal'] == -1, 'close'], label='Sell Signal', marker='v', color='red')
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Price Chart with SMA and Buy/Sell Signals")
st.pyplot(fig)

# Signal Table
st.subheader("Recent Buy/Sell Signals")
st.dataframe(df[['date', 'close', 'sma', 'signal', 'position']].tail(10).style.highlight_max(axis=0))

# Strategy Metrics
st.subheader("Strategy Overview")
st.metric(label="Total Strategy Return", value=f"{total_return:.2f}%")
st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
