import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("SignalEdge — Smarter Market Timing")

# Toggle between live data or uploaded CSV
data_source = st.radio("Select Data Source", ["Live S&P 500", "Upload CSV"])

# SMA + volatility filter setup
sma_period = st.slider("Select SMA Period", 5, 200, 50)
cash_yield = st.number_input("Select Cash Yield (annualized %)", value=2.5)

# Load data
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
else:
    df = yf.download("^GSPC", start="2010-01-01", progress=False)[['Close']].reset_index()
    df.columns = ['date', 'close']

if 'df' in locals():
    df = df[['date', 'close']].dropna()
    df = df.sort_values('date')
    df['sma'] = df['close'].rolling(sma_period).mean()
    df['volatility'] = df['close'].pct_change().rolling(sma_period).std()
    df['signal'] = 0
    df['signal'][sma_period:] = np.where(
        (df['close'][sma_period:] > df['sma'][sma_period:]) &
        (df['volatility'][sma_period:] < df['volatility'][sma_period:].mean()),
        1, -1
    )
    df['position'] = df['signal'].shift().fillna(0)
    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['returns'] * df['position']
    df['cumulative_strategy'] = (1 + df['strategy']).cumprod()
    df['cumulative_benchmark'] = (1 + df['returns']).cumprod()
    df['cumulative_cash'] = (1 + (cash_yield / 100) / 252) ** np.arange(len(df))

    # Metrics
    st.subheader("Strategy Overview")
    total_return = df['cumulative_strategy'].iloc[-1] - 1
    annualized_return = df['strategy'].mean() * 252
    annual_volatility = df['strategy'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - (cash_yield / 100)) / annual_volatility
    max_drawdown = (df['cumulative_strategy'].cummax() - df['cumulative_strategy']).max()

    st.metric("Total Return", f"{total_return:.2%}")
    st.metric("Annualized Return", f"{annualized_return:.2%}")
    st.metric("Volatility", f"{annual_volatility:.2%}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['cumulative_strategy'], label='Strategy')
    ax.plot(df['date'], df['cumulative_benchmark'], label='Buy & Hold')
    ax.plot(df['date'], df['cumulative_cash'], label='Cash Yield')
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please upload a CSV with 'date' and 'close' columns or select Live data.")