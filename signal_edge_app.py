

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Title and Description ---
st.title("SignalEdge — S&P 500 Signal App")
st.markdown("Trend-following + Volatility strategy with optional cash parking and risk metrics.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- Parameters ---
    sma_period = st.slider("Select SMA Period", 5, 200, 50)
    default_cash_yield = st.selectbox("Select Cash Yield (annualized)", [1.0, 2.5, 4.0, 5.0], index=1)

    # --- Calculate SMA ---
    df['sma'] = df['close'].rolling(window=sma_period).mean()

    # --- Generate Signals ---
    df['signal'] = 0
    df.loc[df['close'] > df['sma'], 'signal'] = 1
    df.loc[df['close'] < df['sma'], 'signal'] = -1

    # --- Simulate Position ---
    df['position'] = df['signal'].replace(to_replace=0, method='ffill')
    df['position'].fillna(0, inplace=True)

    # --- Daily Returns & Cash Logic ---
    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    cash_rate_daily = (1 + default_cash_yield / 100) ** (1 / 252) - 1
    df['strategy_return'] = np.where(df['position'].shift(1) == 0, cash_rate_daily, df['strategy_return'])
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    # --- Metrics ---
    total_return = df['cumulative_return'].iloc[-1] - 1
    annualized_return = (df['cumulative_return'].iloc[-1]) ** (252 / len(df)) - 1
    max_drawdown = ((df['cumulative_return'].cummax() - df['cumulative_return']) / df['cumulative_return'].cummax()).max()
    annual_volatility = df['strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.01) / annual_volatility if annual_volatility > 0 else np.nan

    # --- Output ---
    st.subheader("Strategy Overview")
    st.metric("Total Strategy Return", f"{total_return:.2%}")
    st.metric("Annualized Return", f"{annualized_return:.2%}")
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    st.metric("Annual Volatility", f"{annual_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- Chart ---
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['close'], label="Close Price")
    ax.plot(df['date'], df['sma'], label=f"SMA {sma_period}", linestyle="--")
    ax.scatter(df[df['signal'] == 1]['date'], df[df['signal'] == 1]['close'], marker="^", color="green", label="Buy Signal")
    ax.scatter(df[df['signal'] == -1]['date'], df[df['signal'] == -1]['close'], marker="v", color="red", label="Sell Signal")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # --- Show Data Table ---
    st.subheader("Recent Buy/Sell Signals")
    st.dataframe(df[['date', 'close', 'sma', 'signal', 'position']].tail(30))

else:
    st.info("Please upload a CSV file with at least 'date' and 'close' columns.")
