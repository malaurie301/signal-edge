
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv('sp500.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df


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
st.title("SignalEdge — S&P 500 Signal App")

uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])
st.title("SignalEdge — S&P 500 Signal App")

uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
else:
    df = load_data()

sma_period = st.slider("Select SMA Period", min_value=10, max_value=200, value=50)
df = compute_indicators(df, sma_period)

# Layout formatting
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

st.markdown("### Price Chart with SMA and Buy/Sell Signals")
chart_data = df[['close', 'sma']].copy()
st.line_chart(chart_data)

# Show signals (recent rows)
st.markdown("### Recent Buy/Sell Signals")
styled_df = df[['close', 'sma', 'signal', 'position']].tail(10)
st.dataframe(styled_df.style.highlight_max(axis=0))
