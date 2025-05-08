import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Title and logo ---
st.set_page_config(page_title="SignalEdge", layout="wide")
st.title("SignalEdge — S&P 500 Signal App")
st.image("https://raw.githubusercontent.com/malaurie301/signal-edge/main/logo.png", width=200)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv('sp500.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# --- Compute indicators ---
def compute_indicators(df, sma_period):
    df['sma'] = df['close'].rolling(sma_period).mean()
    df['signal'] = 0
    df.loc[df['close'].shift(1) > df['sma'].shift(1), 'signal'] = 1
    df.loc[df['close'].shift(1) < df['sma'].shift(1), 'signal'] = -1
    df['position'] = df['signal'].shift(1)
    df['return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'] * df['return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
else:
    df = load_data()

# --- User input for SMA period ---
sma_period = st.slider("Select SMA Period", min_value=10, max_value=200, value=50)
df = compute_indicators(df, sma_period)

# --- Strategy Overview ---
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

# --- Price Chart with Buy/Sell Signals ---
st.markdown("### Price Chart with SMA and Buy/Sell Signals")
fig, ax = plt.subplots(figsize=(12, 6))
df['close'].plot(ax=ax, label='Close Price')
df['sma'].plot(ax=ax, label=f'SMA {sma_period}')
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
ax.scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green')
ax.scatter(sell_signals.index, sell_signals['close'], label='Sell Signal', marker='v', color='red')
ax.legend()
ax.set_ylabel('Price')
st.pyplot(fig)

# --- Recent signals table ---
st.markdown("### Recent Buy/Sell Signals")
styled_df = df[['close', 'sma', 'signal', 'position']].tail(10)
st.dataframe(styled_df.style.highlight_max(axis=0))
