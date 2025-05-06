
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Dark theme styling
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: white; }
    .stButton button { background-color: #262730; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_sample_data():
    df = pd.read_csv('sp500.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def load_live_data():
    df = yf.download('^GSPC', period='1y')
    df = df[['Close']].rename(columns={'Close': 'close'})
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
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
    df['drawdown'] = (df['cumulative_return'] / df['cumulative_return'].cummax()) - 1
    return df

st.title("SignalEdge — S&P 500 Signal App")

uploaded_file = st.file_uploader("Upload your price data (CSV with 'date' and 'close')", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
else:
    try:
        df = load_live_data()
    except Exception as e:
        st.warning("⚠ Live data unavailable — using sample S&P 500 data.")
        df = load_sample_data()

sma_period = st.slider("Select SMA Period", min_value=10, max_value=200, value=50)
df = compute_indicators(df, sma_period)

# Plot with buy/sell markers
fig, ax = plt.subplots()
ax.plot(df.index, df['close'], label='Close Price')
ax.plot(df.index, df['sma'], label=f'SMA {sma_period}', linestyle='--')
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]
ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=100)
ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=100)
ax.set_title('Price Chart with Buy/Sell Signals')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader("Cumulative Strategy Return")
st.line_chart(df['cumulative_return'])

st.subheader("Recent Signals")
st.write(df[['close', 'sma', 'signal', 'position']].tail(10))

# Performance summary
total_return = df['cumulative_return'].iloc[-1] - 1
sharpe = np.mean(df['strategy_return']) / np.std(df['strategy_return']) * np.sqrt(252)
max_drawdown = df['drawdown'].min()
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Download signal report
csv = df.reset_index()[['date', 'close', 'sma', 'signal', 'position']].to_csv(index=False).encode()
st.download_button("Download Signal Report (CSV)", csv, "signal_report.csv", "text/csv")
