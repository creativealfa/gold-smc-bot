import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold SMC Scalper", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
EAT_TZ = ZoneInfo("Africa/Nairobi")

default_key = st.secrets.get("TWELVE_DATA_API_KEY", "").strip() if hasattr(st, "secrets") else ""

@st.cache_data(ttl=60)
def fetch_gold_series(api_key: str, symbol: str, interval: str) -> tuple[pd.DataFrame, str]:
    if not api_key:
        return pd.DataFrame(), "No API key provided."
    base_url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "timezone": "Africa/Nairobi",
        "apikey": api_key,
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=10)
        if not resp.text.strip():
            return pd.DataFrame(), f"Empty response (HTTP {resp.status_code})"
        res = resp.json()
        if "values" not in res:
            return pd.DataFrame(), res.get("message", f"API Error: {res}")
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_localize(EAT_TZ)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), f"Request failed: {e}"

def find_scalping_setups(df: pd.DataFrame, swing_lookback: int, buf: float, rr: float) -> list[dict]:
    """Scans historical and live candles for Liquidity Sweep + Fair Value Gap (FVG) scalps."""
    signals = []
    if len(df) < swing_lookback + 4:
        return signals

    for i in range(swing_lookback + 2, len(df) - 1):
        prior_window = df.iloc[i - swing_lookback - 2 : i - 2]
        swing_high = float(prior_window["High"].max())
        swing_low = float(prior_window["Low"].min())

        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        ts = df.index[i]

        # SELL SCALP: Wick sweeps prior swing high + closes back below + Bearish FVG
        swept_high = (c1["High"] > swing_high + buf) or (c2["High"] > swing_high + buf)
        closed_below = c3["Close"] < swing_high
        bearish_fvg = c1["Low"] > c3["High"]

        if swept_high and closed_below and bearish_fvg:
            entry = round((c1["Low"] + c3["High"]) / 2, 2)
            sl = round(max(c1["High"], c2["High"]) + 0.80, 2)
            risk = max(sl - entry, 0.50)
            tp = round(entry - (risk * rr), 2)
            signals.append({
                "timestamp": ts,
                "time_str": ts.strftime("%m-%d %H:%M"),
                "side": "SELL",
                "marker_y": float(c3["High"]) + 1.2,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "fvg_top": float(c1["Low"]),
                "fvg_bot": float(c3["High"]),
            })

        # BUY SCALP: Wick sweeps prior swing low + closes back above + Bullish FVG
        swept_low = (c1["Low"] < swing_low - buf) or (c2["Low"] < swing_low - buf)
        closed_above = c3["Close"] > swing_low
        bullish_fvg = c1["High"] < c3["Low"]

        if swept_low and closed_above and bullish_fvg:
            entry = round((c1["High"] + c3["Low"]) / 2, 2)
            sl = round(min(c1["Low"], c2["Low"]) - 0.80, 2)
            risk = max(entry - sl, 0.50)
            tp = round(entry + (risk * rr), 2)
            signals.append({
                "timestamp": ts,
                "time_str": ts.strftime("%m-%d %H:%M"),
                "side": "BUY",
                "marker_y": float(c3["Low"]) - 1.2,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "fvg_top": float(c3["Low"]),
                "fvg_bot": float(c1["High"]),
            })

    return signals

with st.sidebar:
    st.header("Scalper Settings")
    api_key = st.text_input("Twelve Data API Key", type="password", value=default_key).strip()
    timeframe = st.selectbox("Scalping Timeframe", ["5min", "1min", "15min"], index=0)
    swing_len = st.slider("Liquidity Lookback (Candles)", min_value=10, max_value=40, value=20)
    buf = st.number_input("Min Sweep Buffer ($)", value=0.50, step=0.25)
    rr = st.number_input("Risk-to-Reward Ratio", value=2.0, step=0.5)
    if st.button("Refresh Data Now", width="stretch"):
        st.cache_data.clear()

now_eat = datetime.now(EAT_TZ)
st.title("Gold SMC Scalping Terminal")
st.caption(f"Live Liquidity Sweep & FVG Marker | Nairobi Time: {now_eat.strftime('%H:%M EAT')}")

if not api_key:
    st.info("Open the top-left menu (>) and paste your Twelve Data API Key to activate the scalper.")
    st.stop()

df, status_msg = fetch_gold_series(api_key, "XAU/USD", timeframe)
active_symbol = "XAU/USD"
if df.empty:
    df, status_msg = fetch_gold_series(api_key, "PAXG/USD", timeframe)
    active_symbol = "PAXG/USD (Free Spot Gold)"

if df.empty:
    st.error(f"Could not load market data: {status_msg}")
    st.stop()

# Session liquidity levels (Asian & London)
latest_date = df.index[-1].date()
day_df = df[df.index.date == latest_date]
asian_df = day_df.between_time("00:00", "09:59")
london_df = day_df.between_time("10:00", "14:59")

current_price = float(df["Close"].iloc[-1])
london_high = float(london_df["High"].max()) if not london_df.empty else float(day_df["High"].max())
london_low = float(london_df["Low"].min()) if not london_df.empty else float(day_df["Low"].min())

c1, c2, c3 = st.columns(3)
c1.metric(f"Live {active_symbol} ({timeframe})", f"${current_price:,.2f}")
c2.metric("Session High (BSL)", f"${london_high:,.2f}", delta=f"${current_price - london_high:,.2f}")
c3.metric("Session Low (SSL)", f"${london_low:,.2f}", delta=f"${current_price - london_low:,.2f}")

signals = find_scalping_setups(df, swing_len, buf, rr)

# Highlight latest setup banner
if signals:
    latest_sig = signals[-1]
    if latest_sig["side"] == "BUY":
        st.success(f"LATEST SCALP ({latest_sig['time_str']} EAT): **BUY ENTRY @ ${latest_sig['entry']}** | SL: **${latest_sig['sl']}** | TP: **${latest_sig['tp']}**")
    else:
        st.error(f"LATEST SCALP ({latest_sig['time_str']} EAT): **SELL ENTRY @ ${latest_sig['entry']}** | SL: **${latest_sig['sl']}** | TP: **${latest_sig['tp']}**")
else:
    st.info("Scanning for liquidity sweep + FVG scalping entries...")

# Build Candlestick Chart with BUY/SELL Markers
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name=active_symbol
)])

buy_sigs = [s for s in signals if s["side"] == "BUY"]
sell_sigs = [s for s in signals if s["side"] == "SELL"]

if buy_sigs:
    fig.add_trace(go.Scatter(
        x=[s["timestamp"] for s in buy_sigs],
        y=[s["marker_y"] for s in buy_sigs],
        mode="markers+text",
        name="BUY Entry",
        text=["BUY" for _ in buy_sigs],
        textposition="bottom center",
        textfont=dict(color="#00ff88", size=11),
        marker=dict(symbol="triangle-up", size=14, color="#00ff88", line=dict(width=1, color="black"))
    ))

if sell_sigs:
    fig.add_trace(go.Scatter(
        x=[s["timestamp"] for s in sell_sigs],
        y=[s["marker_y"] for s in sell_sigs],
        mode="markers+text",
        name="SELL Entry",
        text=["SELL" for _ in sell_sigs],
        textposition="top center",
        textfont=dict(color="#ff4b4b", size=11),
        marker=dict(symbol="triangle-down", size=14, color="#ff4b4b", line=dict(width=1, color="black"))
    ))

# Draw Entry, SL, and TP lines for the most recent scalp setup
if signals:
    last = signals[-1]
    fig.add_hline(y=last["entry"], line_dash="dot", line_color="#38bdf8", annotation_text=f"Entry ${last['entry']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

fig.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, width="stretch")

if signals:
    st.subheader("Recent Scalping Signals")
    hist_df = pd.DataFrame(signals)[["time_str", "side", "entry", "sl", "tp"]].iloc[::-1]
    hist_df.columns = ["Time (EAT)", "Signal", "FVG Entry ($)", "Stop Loss ($)", "Take Profit ($)"]
    st.dataframe(hist_df, hide_index=True, width="stretch")
