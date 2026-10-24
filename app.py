import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import ta

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="AI Trading Bot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Trading Bot Pro")
st.caption("Paper Trading Mode • Educational Purpose Only • High Risk Warning")

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Configuration")

symbol = st.sidebar.selectbox(
    "Trading Pair",
    ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT"]
)

timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

strategy = st.sidebar.selectbox("Strategy", [
    "Combined AI Score",
    "MA Crossover",
    "RSI Reversal",
    "Momentum"
])

st.sidebar.subheader("Risk Management")
capital = st.sidebar.number_input("Starting Capital (USDT)", value=10000.0, step=500.0, min_value=100.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.5, 0.1)
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 8.0, 2.5, 0.1)
take_profit_pct = st.sidebar.slider("Take Profit (%)", 2.0, 15.0, 5.0, 0.1)
max_daily_loss = st.sidebar.slider("Max Daily Loss (%)", 2.0, 15.0, 5.0, 0.5)

auto_trade = st.sidebar.toggle("🔄 Enable Auto-Trading", value=False)

st.sidebar.markdown("---")
st.sidebar.error(
    "⚠️ **Disclaimer**\n\n"
    "This is a **paper-trading** educational bot only.\n\n"
    "Real trading involves substantial risk of loss.\n"
    "Never risk money you cannot afford to lose.\n"
    "This is not financial advice."
)

# ====================== SESSION STATE ======================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "cash": float(capital),
        "position": 0.0,
        "entry_price": 0.0,
        "trades": [],
        "daily_pnl": 0.0,
        "last_reset": datetime.now().date()
    }

# ====================== EXCHANGE & DATA ======================
@st.cache_resource
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})

exchange = get_exchange()

@st.cache_data(ttl=45)
def fetch_data(symbol: str, timeframe: str, limit: int = 250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=9)
    df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["momentum"] = df["close"].pct_change(6)
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)
    return df

def generate_signal(df: pd.DataFrame, strategy_name: str):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    reasons = []

    ma_bull = latest["sma_fast"] > latest["sma_slow"] and prev["sma_fast"] <= prev["sma_slow"]
    ma_bear = latest["sma_fast"] < latest["sma_slow"] and prev["sma_fast"] >= prev["sma_slow"]
    rsi_oversold = latest["rsi"] < 32
    rsi_overbought = latest["rsi"] > 68
    mom_up = latest["momentum"] > 0.012
    mom_down = latest["momentum"] < -0.012

    if strategy_name == "MA Crossover":
        if ma_bull:
            score = 3.0
            reasons.append("Bullish MA Crossover")
        elif ma_bear:
            score = -3.0
            reasons.append("Bearish MA Crossover")

    elif strategy_name == "RSI Reversal":
        if rsi_oversold:
            score = 2.5
            reasons.append("RSI Oversold")
        elif rsi_overbought:
            score = -2.5
            reasons.append("RSI Overbought")

    elif strategy_name == "Momentum":
        if mom_up:
            score = 2.0
            reasons.append("Strong Positive Momentum")
        elif mom_down:
            score = -2.0
            reasons.append("Strong Negative Momentum")

    else:  # Combined AI Score
        if ma_bull:
            score += 2.0
            reasons.append("MA Bullish Cross")
        if ma_bear:
            score -= 2.0
            reasons.append("MA Bearish Cross")
        if rsi_oversold:
            score += 1.5
            reasons.append("RSI Oversold")
        if rsi_overbought:
            score -= 1.5
            reasons.append("RSI Overbought")
        if mom_up:
            score += 1.0
            reasons.append("Positive Momentum")
        if mom_down:
            score -= 1.0
            reasons.append("Negative Momentum")

    if score >= 2.0:
        signal = "BUY"
    elif score <= -2.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, reasons, latest

# ====================== MAIN APP ======================
df = fetch_data(symbol, timeframe)

if df is not None:
    df = add_indicators(df)
    signal, score, reasons, latest = generate_signal(df, strategy)
    current_price = float(latest["close"])

    # Reset daily PnL on new day
    if st.session_state.portfolio["last_reset"] != datetime.now().date():
        st.session_state.portfolio["daily_pnl"] = 0.0
        st.session_state.portfolio["last_reset"] = datetime.now().date()

    # ====================== TOP METRICS ======================
    equity = st.session_state.portfolio["cash"] + st.session_state.portfolio["position"] * current_price
    unrealized = 0.0
    if st.session_state.portfolio["position"] > 0:
        unrealized = (current_price - st.session_state.portfolio["entry_price"]) * st.session_state.portfolio["position"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Price", f"${current_price:,.2f}")
    m2.metric("Signal", signal, delta=f"Score {score:.1f}")
    m3.metric("RSI (14)", f"{latest['rsi']:.1f}")
    m4.metric("Total Equity", f"${equity:,.2f}")
    m5.metric("Unrealized PnL", f"${unrealized:,.2f}")

    # ====================== CHART ======================
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["sma_fast"],
        name="SMA Fast (9)", line=dict(width=1.3, color="#00f5d4")
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["sma_slow"],
        name="SMA Slow (21)", line=dict(width=1.3, color="#ff9f1c")
    ))
    fig.update_layout(
        title=f"{symbol}  •  {timeframe}  •  Strategy: {strategy}",
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====================== SIGNAL ANALYSIS ======================
    st.subheader("🧠 AI Signal Analysis")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown(f"### Signal: `{signal}`")
        st.markdown(f"**Score:** `{score:.2f}`")
    with col_s2:
        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("• No strong confluence detected → HOLD")

    # ====================== TRADING FUNCTIONS ======================
    def execute_buy():
        if st.session_state.portfolio["position"] > 0:
            return False, "Already in a position"
        if st.session_state.portfolio["daily_pnl"] <= -(capital * max_daily_loss / 100):
            return False, "Max daily loss limit reached"

        risk_amount = st.session_state.portfolio["cash"] * (risk_per_trade / 100)
        size = risk_amount / current_price
        cost = size * current_price

        if cost > st.session_state.portfolio["cash"]:
            return False, "Insufficient cash"

        st.session_state.portfolio["position"] = size
        st.session_state.portfolio["entry_price"] = current_price
        st.session_state.portfolio["cash"] -= cost

        st.session_state.portfolio["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "BUY",
            "price": round(current_price, 4),
            "size": round(size, 6),
            "pnl": 0.0,
            "reason": "Manual / Auto"
        })
        return True, f"Bought {size:.6f} @ ${current_price:.2f}"

    def execute_sell(reason="Manual"):
        if st.session_state.portfolio["position"] == 0:
            return False, "No open position"

        size = st.session_state.portfolio["position"]
        pnl = (current_price - st.session_state.portfolio["entry_price"]) * size
        st.session_state.portfolio["cash"] += size * current_price
        st.session_state.portfolio["daily_pnl"] += pnl

        st.session_state.portfolio["trades"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": "SELL",
            "price": round(current_price, 4),
            "size": round(size, 6),
            "pnl": round(pnl, 2),
            "reason": reason
        })

        st.session_state.portfolio["position"] = 0.0
        st.session_state.portfolio["entry_price"] = 0.0
        return True, f"Sold. PnL: ${pnl:.2f} ({reason})"

    # Auto Stop-Loss / Take-Profit check
    if st.session_state.portfolio["position"] > 0:
        entry = st.session_state.portfolio["entry_price"]
        sl_price = entry * (1 - stop_loss_pct / 100)
        tp_price = entry * (1 + take_profit_pct / 100)

        if current_price <= sl_price:
            success, msg = execute_sell("Stop-Loss")
            if success:
                st.warning(msg)
        elif current_price >= tp_price:
            success, msg = execute_sell("Take-Profit")
            if success:
                st.success(msg)

    # Auto-Trading logic
    if auto_trade:
        if signal == "BUY" and st.session_state.portfolio["position"] == 0:
            success, msg = execute_buy()
            if success:
                st.success(f"🤖 Auto-Trade: {msg}")
        elif signal == "SELL" and st.session_state.portfolio["position"] > 0:
            success, msg = execute_sell("AI Signal")
            if success:
                st.success(f"🤖 Auto-Trade: {msg}")

    # ====================== MANUAL CONTROLS ======================
    st.subheader("🎮 Manual Controls")
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("🟢 BUY", use_container_width=True, type="primary"):
            success, msg = execute_buy()
            if success:
                st.success(msg)
            else:
                st.warning(msg)

    with b2:
        if st.button("🔴 SELL", use_container_width=True):
            success, msg = execute_sell("Manual")
            if success:
                st.success(msg)
            else:
                st.warning(msg)

    with b3:
        if st.button("🔄 Reset Portfolio", use_container_width=True):
            st.session_state.portfolio = {
                "cash": float(capital),
                "position": 0.0,
                "entry_price": 0.0,
                "trades": [],
                "daily_pnl": 0.0,
                "last_reset": datetime.now().date()
            }
            st.success("Portfolio has been reset")

    # ====================== PORTFOLIO ======================
    st.subheader("💼 Portfolio Overview")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Available Cash", f"${st.session_state.portfolio['cash']:,.2f}")
    p2.metric("Position Size", f"{st.session_state.portfolio['position']:.6f}")
    p3.metric("Entry Price", f"${st.session_state.portfolio['entry_price']:,.2f}" if st.session_state.portfolio['entry_price'] > 0 else "—")
    p4.metric("Daily PnL", f"${st.session_state.portfolio['daily_pnl']:,.2f}")

    # Trade History
    if st.session_state.portfolio["trades"]:
        st.subheader("📜 Trade History")
        trades_df = pd.DataFrame(st.session_state.portfolio["trades"])
        st.dataframe(trades_df, use_container_width=True, height=280)

else:
    st.warning("Unable to fetch market data. Please check your internet connection or try again later.")

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Built for educational purposes only • Paper trading • Not financial advice • Use at your own risk")
```