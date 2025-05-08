import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("SignalEdge — S&P 500 Signal App")
st.markdown("**Trend-following + Volatility strategy with optional cash yield and benchmark**")

uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    sma_period = st.slider("Select SMA Period", 5, 200, 50)
    cash_yield = st.number_input("Select Cash Yield (annualized)", value=2.0)

    df['sma'] = df['close'].rolling(window=sma_period).mean()
    df['signal'] = 0
    df['signal'][sma_period:] = np.where(df['close'][sma_period:] > df['sma'][sma_period:], 1, -1)
    df['position'] = df['signal'].shift(1)

    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['position'] * df['returns']
    df['cumulative_strategy'] = (1 + df['strategy']).cumprod()
    df['cumulative_benchmark'] = (1 + df['returns']).cumprod()

    if not df.empty and "cumulative_strategy" in df.columns:
        total_return = df["cumulative_strategy"].iloc[-1] - 1
    else:
        total_return = 0

    if not df.empty and "cumulative_benchmark" in df.columns:
        benchmark_return = df["cumulative_benchmark"].iloc[-1] - 1
    else:
        benchmark_return = 0

    # Optional: Adjust for cash yield if not in market
    df['strategy_cash'] = np.where(df['position'] == 0, cash_yield / 252 / 100, df['strategy'])
    df['cumulative_strategy_cash'] = (1 + df['strategy_cash']).cumprod()

    annualized_return = df['strategy'].mean() * 252
    annual_volatility = df['strategy'].std() * np.sqrt(252)
    max_drawdown = (df['cumulative_strategy'] / df['cumulative_strategy'].cummax() - 1).min()

    sharpe_ratio = (annualized_return - (cash_yield / 100)) / annual_volatility if annual_volatility else 0

    st.subheader("Strategy Overview")
    st.metric("Total Strategy Return", f"{total_return:.2%}")
    st.metric("Annualized Return", f"{annualized_return:.2%}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    st.metric("Annual Volatility", f"{annual_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    fig, ax = plt.subplots()
    ax.plot(df['date'], df['close'], label='Close Price')
    ax.plot(df['date'], df['sma'], label=f'SMA {sma_period}', linestyle='--')
    ax.scatter(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], marker='^', color='green', label='Buy Signal')
    ax.scatter(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], marker='v', color='red', label='Sell Signal')
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.subheader("Recent Buy/Sell Signals")
    st.dataframe(df[['date', 'close', 'sma', 'signal', 'position']].tail(20))
else:
    st.info("Please upload a CSV file with at least 'date' and 'close' columns.")
