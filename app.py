import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Simple AI + Order Flow Terminal", page_icon="⚡", layout="wide")
EAT = ZoneInfo("Africa/Nairobi")
st.title("Simple AI + CNX Order Flow Gold Terminal")
st.caption(f"ATR AI Trend + Order Flow Delta + FVG Zones + Auto-Journal • Nairobi Time: {datetime.now(EAT).strftime('%H:%M EAT')}")

try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

api_key = st.text_input("Twelve Data API Key", value=saved_key, type="password").strip()
symbol = st.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
tf = st.selectbox("Timeframe", ["5min", "15min", "1min"], index=0)
atr_mult = st.slider("Simple AI Sensitivity (ATR Multiplier)", 1.5, 3.5, 2.2, step=0.1)

if st.button("Refresh Live Candles & Order Flow"):
    st.cache_data.clear()

if not api_key:
    st.warning("Paste your Twelve Data API Key above (or save it in Streamlit Secrets) to start.")
    st.stop()

@st.cache_data(ttl=60)
def get_candles(key: str, sym: str, interval: str):
    url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": sym, "interval": interval, "outputsize": 200, "timezone": "Africa/Nairobi", "apikey": key}
    try:
        res = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(4, 7)).json()
        if "values" not in res:
            return pd.DataFrame(), res.get("message", str(res))
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_localize(EAT)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)

df, status = get_candles(api_key, symbol, tf)
if df.empty:
    st.error(f"Data error ({symbol}): {status}")
    st.stop()

# 1. Simple Trade AI: ATR Trailing Stop + 50 EMA + RSI
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill()
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
df["rsi"] = (100 - (100 / (1 + (gain / loss)))).fillna(50)

hl2 = (df["High"] + df["Low"]) / 2
up_band = (hl2 - atr_mult * df["atr"]).to_numpy()
dn_band = (hl2 + atr_mult * df["atr"]).to_numpy()
close_arr = df["Close"].to_numpy()
trend_dir = np.ones(len(df), dtype=int)
trail_stop = np.zeros(len(df))

for i in range(1, len(df)):
    if close_arr[i - 1] > up_band[i - 1]:
        up_band[i] = max(up_band[i], up_band[i - 1])
    if close_arr[i - 1] < dn_band[i - 1]:
        dn_band[i] = min(dn_band[i], dn_band[i - 1])
    if close_arr[i] > dn_band[i - 1]:
        trend_dir[i] = 1
    elif close_arr[i] < up_band[i - 1]:
        trend_dir[i] = -1
    else:
        trend_dir[i] = trend_dir[i - 1]
    trail_stop[i] = up_band[i] if trend_dir[i] == 1 else dn_band[i]

df["trend_dir"] = trend_dir
df["trail_stop"] = trail_stop

# 2. CNX Order Flow: Internal Candle Delta & FVG Imbalances
rng = (df["High"] - df["Low"]).replace(0, 0.01)
df["net_delta"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng

# 3. Fused Signal Generator + Auto Win/Loss Journal
journal = []
last_sig = 0

for i in range(20, len(df)):
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    kz = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)

    bull_flip = (df["trend_dir"].iloc[i] == 1) and (df["trend_dir"].iloc[i - 1] == -1)
    bull_fvg = (c3["Low"] > c1["High"]) and (c2["Close"] > c2["Open"])
    bull_flow = (df["trend_dir"].iloc[i] == 1) and bull_fvg and (c3["net_delta"] > 0.15) and (c3["rsi"] < 68) and (c3["Close"] > c3["ema50"])

    bear_flip = (df["trend_dir"].iloc[i] == -1) and (df["trend_dir"].iloc[i - 1] == 1)
    bear_fvg = (c3["High"] < c1["Low"]) and (c2["Close"] < c2["Open"])
    bear_flow = (df["trend_dir"].iloc[i] == -1) and bear_fvg and (c3["net_delta"] < -0.15) and (c3["rsi"] > 32) and (c3["Close"] < c3["ema50"])

    trade = None
    if ((bull_flip and c3["net_delta"] > 0) or bull_flow) and last_sig != 1:
        last_sig = 1
        entry = round(float(c3["Close"]), 2)
        sl = round(float(c3["trail_stop"] - c3["atr"] * 0.3), 2)
        tp = round(entry + max(entry - sl, float(c3["atr"])) * 2.0, 2)
        conf = 70 + (15 if kz else 0) + (15 if bull_fvg else 0)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "BUY", "y": float(c3["Low"] - c3["atr"] * 0.6), "entry": entry, "sl": sl, "tp": tp, "conf": conf}

    elif ((bear_flip and c3["net_delta"] < 0) or bear_flow) and last_sig != -1:
        last_sig = -1
        entry = round(float(c3["Close"]), 2)
        sl = round(float(c3["trail_stop"] + c3["atr"] * 0.3), 2)
        tp = round(entry - max(sl - entry, float(c3["atr"])) * 2.0, 2)
        conf = 70 + (15 if kz else 0) + (15 if bear_fvg else 0)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "SELL", "y": float(c3["High"] + c3["atr"] * 0.6), "entry": entry, "sl": sl, "tp": tp, "conf": conf}

    if trade:
        res_tag, pnl = "OPEN", 0.0
        for idx_f, bar in df.iloc[i + 1 :].iterrows():
            hit_sl = bar["Low"] <= trade["sl"] if trade["side"] == "BUY" else bar["High"] >= trade["sl"]
            hit_tp = bar["High"] >= trade["tp"] if trade["side"] == "BUY" else bar["Low"] <= trade["tp"]
            if hit_sl:
                res_tag, pnl = "LOSS", -1.0
                break
            if hit_tp:
                res_tag, pnl = "WIN", 2.0
                break
        trade.update({"result": res_tag, "pnl": pnl})
        journal.append(trade)

wins = len([t for t in journal if t["result"] == "WIN"])
losses = len([t for t in journal if t["result"] == "LOSS"])
net_r = sum(t["pnl"] for t in journal)
flow_bias = "BULLISH ORDER FLOW (+)" if df["net_delta"].iloc[-1] > 0 else "BEARISH ORDER FLOW (-)"

st.metric(f"Live {symbol} • {flow_bias} • Journal: {wins}W-{losses}L ({net_r:+.1f}R)", f"${float(df['Close'].iloc[-1]):,.2f}")

if journal:
    last = journal[-1]
    msg = f"LATEST SIGNAL ({last['time']} EAT): **{last['side']} AI** [{last['result']}] | Conf: **{last['conf']}%** | Entry: **${last['entry']}** | SL: **${last['sl']}** | TP: **${last['tp']}**"
    st.success(msg) if last["side"] == "BUY" else st.error(msg)

# Plot Candlesticks + Simple AI Green/Red Trend Line + Order Flow Markers
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))

bull_line = np.where(df["trend_dir"] == 1, df["trail_stop"], np.nan)
bear_line = np.where(df["trend_dir"] == -1, df["trail_stop"], np.nan)
fig.add_trace(go.Scatter(x=df.index, y=bull_line, mode="lines", name="Simple AI Bull Trend", line=dict(color="#00ff88", width=2.2)))
fig.add_trace(go.Scatter(x=df.index, y=bear_line, mode="lines", name="Simple AI Bear Trend", line=dict(color="#ff4b4b", width=2.2)))

for side, col, sym, pos in [("BUY", "#00ff88", "triangle-up", "bottom center"), ("SELL", "#ff4b4b", "triangle-down", "top center")]:
    pts = [t for t in journal if t["side"] == side]
    if pts:
        fig.add_trace(go.Scatter(x=[p["ts"] for p in pts], y=[p["y"] for p in pts], mode="markers+text", name=side, text=[f"{side} AI ({p['result']})" for p in pts], textposition=pos, textfont=dict(color=col, size=10), marker=dict(symbol=sym, size=14, color=col)))

if journal:
    last = journal[-1]
    fig.add_hline(y=last["entry"], line_color="#38bdf8", annotation_text=f"Entry ${last['entry']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

if journal:
    st.subheader("Simple AI + Order Flow Journal")
    j_df = pd.DataFrame([{"Time (EAT)": t["time"], "Signal": f"{t['side']} AI", "Outcome": t["result"], "P&L": f"{t['pnl']:+.1f}R", "Conf": f"{t['conf']}%", "Entry": f"${t['entry']}", "SL": f"${t['sl']}", "TP": f"${t['tp']}"} for t in reversed(journal)])
    st.dataframe(j_df, hide_index=True, use_container_width=True)
