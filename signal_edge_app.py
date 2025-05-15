import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.set_page_config(page_title="SignalEdge — S&P 500 Signal App", layout="wide")

# App title
st.title("SignalEdge — S&P 500 Signal App")
st.markdown("_Trend-following + Volatility strategy with optional cash yield and benchmark comparison._")

# Data source toggle
data_source = st.radio("Select Data Source:", ["Upload CSV", "Live Data"])

# Load data
df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
elif data_source == "Live Data":
    st.write("Using live S&P 500 data from Yahoo Finance...")
    df = yf.download('^GSPC', period='1y', progress=False)
    df = df[['Close']].rename(columns={'Close': 'close'})
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)

if df is not None:
    sma_period = st.slider("Select SMA Period", 5, 200, 50)
    cash_yield = st.number_input("Select Cash Yield (annualized)", min_value=0.0, max_value=10.0, value=2.5)

    df['sma'] = df['close'].rolling(window=sma_period).mean()
    df['signal'] = 0
    df.loc[df.index[sma_period:], 'signal'] = np.where(
    df['close'][sma_period:] > df['sma'][sma_period:], 1, -1
)
    df['position'] = df['signal'].shift(1)
    df.dropna(inplace=True)

    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['returns'] * df['position']
    df['cumulative_strategy'] = (1 + df['strategy'].fillna(0)).cumprod()
    df['cumulative_benchmark'] = (1 + df['returns'].fillna(0)).cumprod()

    total_return = df['cumulative_strategy'].iloc[-1] - 1
    annualized_return = (df['cumulative_strategy'].iloc[-1]) ** (252 / len(df)) - 1
    max_drawdown = (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min()
    annual_volatility = df['strategy'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - (cash_yield / 100)) / annual_volatility

    st.subheader("Price Chart with Signals")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['date'], df['close'], label='Close Price', linewidth=2)
    ax.plot(df['date'], df['sma'], label=f'SMA {sma_period}', linestyle='--')

    ax.plot(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], '^', markersize=10, color='green', label='Buy Signal')
    ax.plot(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], 'v', markersize=10, color='red', label='Sell Signal')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Strategy Overview")
    st.write(f"**Total Strategy Return:** {total_return:.2%}")
    st.write(f"**Annualized Return:** {annualized_return:.2%}")
    st.write(f"**Max Drawdown:** {max_drawdown:.2%}")
    st.write(f"**Annual Volatility:** {annual_volatility:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
