import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold SMC Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
EAT_TZ = ZoneInfo("Africa/Nairobi")

default_key = st.secrets.get("TWELVE_DATA_API_KEY", "").strip() if hasattr(st, "secrets") else ""

@st.cache_data(ttl=60)
def fetch_gold_series(api_key: str, symbol: str) -> tuple[pd.DataFrame, str]:
    if not api_key:
        return pd.DataFrame(), "No API key provided."
    url = (
        f"https://twelvedata.com"
        f"?symbol={symbol}&interval=5min&outputsize=180"
        f"&timezone=Africa/Nairobi&apikey={api_key}"
    )
    try:
        res = requests.get(url, timeout=10).json()
        if "values" not in res:
            return pd.DataFrame(), res.get("message", "Unknown API response")
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_localize(EAT_TZ)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)

def scan_overlap_signal(df: pd.DataFrame, l_high: float, l_low: float, target_date, buf: float, rr: float):
    ov = df[df.index.date == target_date].between_time("15:00", "18:59")
    if len(ov) < 4:
        return None
    c1, c2, c3 = ov.iloc[-4], ov.iloc[-3], ov.iloc[-2]
    ts = ov.index[-2].strftime("%H:%M EAT")
    if ((c1["High"] > l_high + buf) or (c2["High"] > l_high + buf)) and (c3["Close"] < l_high) and (c1["Low"] > c3["High"]):
        entry = round((c1["Low"] + c3["High"]) / 2, 2)
        sl = round(max(c1["High"], c2["High"]) + 1.0, 2)
        tp = round(entry - (sl - entry) * rr, 2)
        return {"type": "SELL (Bearish Liquidity Sweep)", "time": ts, "entry": entry, "sl": sl, "tp": tp}
    if ((c1["Low"] < l_low - buf) or (c2["Low"] < l_low - buf)) and (c3["Close"] > l_low) and (c1["High"] < c3["Low"]):
        entry = round((c1["High"] + c3["Low"]) / 2, 2)
        sl = round(min(c1["Low"], c2["Low"]) - 1.0, 2)
        tp = round(entry + (entry - sl) * rr, 2)
        return {"type": "BUY (Bullish Liquidity Sweep)", "time": ts, "entry": entry, "sl": sl, "tp": tp}
    return None

with st.sidebar:
    st.header("Bot Settings")
    api_key = st.text_input("Twelve Data API Key", type="password", value=default_key).strip()
    symbol_choice = st.selectbox("Gold Feed", ["Auto (XAU/USD -> Free PAXG/USD)", "PAXG/USD (Free Spot Gold)", "XAU/USD (Paid Plan)"])
    buf = st.number_input("Min Sweep Buffer ($)", value=1.50, step=0.25)
    rr = st.number_input("Risk-to-Reward Ratio", value=2.5, step=0.5)
    if st.button("Refresh Data Now", width="stretch"):
        st.cache_data.clear()

now_eat = datetime.now(EAT_TZ)
st.title("Gold Liquidity Hunter (SMC)")
st.caption(f"Live Session Tracker | Nairobi Time: {now_eat.strftime('%H:%M EAT')}")

if not api_key:
    st.info("Open the top-left menu (>) and paste your Twelve Data API Key to load the live Gold chart.")
    st.stop()

active_symbol = "XAU/USD" if "XAU" in symbol_choice and "Auto" not in symbol_choice else "PAXG/USD"
if "Auto" in symbol_choice:
    df, status_msg = fetch_gold_series(api_key, "XAU/USD")
    if df.empty:
        df, status_msg = fetch_gold_series(api_key, "PAXG/USD")
        active_symbol = "PAXG/USD (Free Spot Gold Proxy)"
    else:
        active_symbol = "XAU/USD"
else:
    df, status_msg = fetch_gold_series(api_key, active_symbol)

if df.empty:
    st.error(f"Could not load market data: {status_msg}")
    st.info("Double-check that your Twelve Data API key has no extra spaces and click 'Refresh Data Now' in the sidebar.")
    st.stop()

latest_date = df.index[-1].date()
london_df = df[df.index.date == latest_date].between_time("10:00", "14:59")
session_tag, active_date = "Today London", latest_date
if london_df.empty and len(df.index.normalize().unique()) > 1:
    prev_date = df.index.normalize().unique()[-2].date()
    london_df = df[df.index.date == prev_date].between_time("10:00", "14:59")
    session_tag, active_date = f"Prev London ({prev_date})", prev_date

current_price = df["Close"].iloc[-1]
london_high = float(london_df["High"].max()) if not london_df.empty else current_price
london_low = float(london_df["Low"].min()) if not london_df.empty else current_price

st.caption(f"Active Feed: **{active_symbol}**")
c1, c2, c3 = st.columns(3)
c1.metric("Live Gold Price", f"${current_price:,.2f}")
c2.metric(f"{session_tag} High (BSL)", f"${london_high:,.2f}", delta=f"${current_price - london_high:,.2f}")
c3.metric(f"{session_tag} Low (SSL)", f"${london_low:,.2f}", delta=f"${current_price - london_low:,.2f}")

sig = scan_overlap_signal(df, london_high, london_low, active_date, buf, rr)
if sig:
    st.success(f"ACTIVE OVERLAP SIGNAL ({sig['time']}): **{sig['type']}** | 50% FVG Entry: **${sig['entry']}** | SL: **${sig['sl']}** | TP: **${sig['tp']}**")
else:
    st.warning("Scanning London-NY Overlap (15:00 - 19:00 EAT): No confirmed sweep + FVG setup yet.")

fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=active_symbol)])
fig.add_hline(y=london_high, line_dash="dash", line_color="#ff4b4b", annotation_text="London High (BSL)")
fig.add_hline(y=london_low, line_dash="dash", line_color="#00cc96", annotation_text="London Low (SSL)")
fig.update_layout(template="plotly_dark", height=460, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")
