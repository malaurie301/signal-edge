import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load default data
@st.cache_data
def load_data():
    df = pd.read_csv('sp500.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# Compute indicators
def compute_indicators(df, sma_period):
    df['sma'] = df['close'].rolling(sma_period).mean()
    df['signal'] = 0
    df.loc[df['close'] > df['sma'], 'signal'] = 1
    df.loc[df['close'] < df['sma'], 'signal'] = -1
    df['position'] = df['signal'].shift(1)
    df['return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'] * df['return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

# App UI
st.set_page_config(page_title="SignalEdge – S&P 500 Signal App", layout="wide")
st.title("SignalEdge — S&P 500 Signal App")

uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
else:
    df = load_data()

# Strategy settings
sma_period = st.slider("Select SMA Period", min_value=10, max_value=200, value=50)
df = compute_indicators(df, sma_period)

# Strategy overview
st.markdown("### Strategy Overview")
col1, col2 = st.columns(2)
with col1:
    total_return = df['cumulative_return'].iloc[-1] - 1
    st.metric("Total Strategy Return", f"{total_return:.2%}")
with col2:
    if 'strategy_return' in df.columns and not df['strategy_return'].empty:
        sharpe = np.mean(df['strategy_return']) / np.std(df['strategy_return'])
    else:
        sharpe = np.nan
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Price chart with arrows
st.markdown("### Price Chart with SMA and Buy/Sell Signals")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['close'], label='Close Price')
ax.plot(df.index, df['sma'], label=f'SMA {sma_period}', linestyle='--')

# Plot arrows
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
ax.scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green', s=100)
ax.scatter(sell_signals.index, sell_signals['close'], label='Sell Signal', marker='v', color='red', s=100)

ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Price')
st.pyplot(fig)

# Recent signals table
st.markdown("### Recent Buy/Sell Signals")
signal_table = df[['close', 'sma', 'signal', 'position']].tail(10)
st.dataframe(signal_table.style.highlight_max(axis=0))
 
