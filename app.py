import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold High-Accuracy AI Terminal", page_icon="🎯", layout="wide")
EAT = ZoneInfo("Africa/Nairobi")
st.title("Gold High-Accuracy AI & Order Flow Terminal")
st.caption(f"Pullback Entry + Macro Trend + Killzone Filter + Auto-Journal • Nairobi Time: {datetime.now(EAT).strftime('%H:%M EAT')}")

try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

api_key = st.text_input("Twelve Data API Key", value=saved_key, type="password").strip()
symbol = st.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
tf = st.selectbox("Timeframe (15min Recommended)", ["15min", "5min"], index=0)
killzone_only = st.checkbox("Only Trade High-Volume Killzones (10:00-14:00 & 15:00-20:00 EAT)", value=True)
rr_target = st.slider("Take Profit Target (Risk-to-Reward)", 1.5, 3.0, 1.8, step=0.1)

if st.button("Refresh Live Candles & Recalculate Win Rate"):
    st.cache_data.clear()

if not api_key:
    st.warning("Paste your Twelve Data API Key above (or save it in Streamlit Secrets) to start.")
    st.stop()

@st.cache_data(ttl=60)
def get_candles(key: str, sym: str, interval: str):
    url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": sym, "interval": interval, "outputsize": 250, "timezone": "Africa/Nairobi", "apikey": key}
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

# 1. Macro Trend EMAs, ATR & RSI
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill().ffill().fillna(1.0)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
df["rsi"] = (100 - (100 / (1 + (gain / loss)))).fillna(50)

# 2. ATR Trailing Trend (Sensitivity 2.5 for cleaner swings)
hl2 = (df["High"] + df["Low"]) / 2.0
up_band = (hl2 - 2.5 * df["atr"]).to_numpy(dtype=float, copy=True)
dn_band = (hl2 + 2.5 * df["atr"]).to_numpy(dtype=float, copy=True)
close_arr = df["Close"].to_numpy(dtype=float, copy=True)
trend_dir = np.ones(len(df), dtype=int)
trail_stop = up_band.copy()

for i in range(1, len(df)):
    if close_arr[i - 1] > up_band[i - 1]:
        up_band[i] = float(np.maximum(up_band[i], up_band[i - 1]))
    if close_arr[i - 1] < dn_band[i - 1]:
        dn_band[i] = float(np.minimum(dn_band[i], dn_band[i - 1]))
    if close_arr[i] > dn_band[i - 1]:
        trend_dir[i] = 1
    elif close_arr[i] < up_band[i - 1]:
        trend_dir[i] = -1
    else:
        trend_dir[i] = trend_dir[i - 1]
    trail_stop[i] = up_band[i] if trend_dir[i] == 1 else dn_band[i]

df["trend_dir"] = trend_dir
df["trail_stop"] = trail_stop

# 3. Order Flow Candle Delta
rng = (df["High"] - df["Low"]).replace(0, 0.01)
df["net_delta"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng

# 4. Pullback Limit Entry Scanner + Honest Forward Journal
journal = []
cooldown = 0

for i in range(30, len(df)):
    if cooldown > 0:
        cooldown -= 1
        continue

    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    in_kz = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)
    if killzone_only and not in_kz:
        continue

    atr = float(c3["atr"])
    macro_bull = (c3["Close"] > c3["ema50"]) and (c3["ema50"] > c3["ema100"])
    macro_bear = (c3["Close"] < c3["ema50"]) and (c3["ema50"] < c3["ema100"])

    # Strong bullish displacement + positive order flow in confirmed macro uptrend
    bull_setup = macro_bull and (c3["trend_dir"] == 1) and (c3["net_delta"] > 0.25) and (45 <= c3["rsi"] <= 66) and (c3["Close"] > c2["High"])
    # Strong bearish displacement + negative order flow in confirmed macro downtrend
    bear_setup = macro_bear and (c3["trend_dir"] == -1) and (c3["net_delta"] < -0.25) and (34 <= c3["rsi"] <= 55) and (c3["Close"] < c2["Low"])

    trade = None
    if bull_setup:
        cooldown = 6
        entry = round(float(c3["Close"] - 0.35 * atr), 2)
        sl = round(entry - 1.25 * atr, 2)
        tp = round(entry + (1.25 * atr) * rr_target, 2)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "BUY", "y": float(c3["Low"] - 0.6 * atr), "entry": entry, "sl": sl, "tp": tp}
    elif bear_setup:
        cooldown = 6
        entry = round(float(c3["Close"] + 0.35 * atr), 2)
        sl = round(entry + 1.25 * atr, 2)
        tp = round(entry - (1.25 * atr) * rr_target, 2)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "SELL", "y": float(c3["High"] + 0.6 * atr), "entry": entry, "sl": sl, "tp": tp}

    if trade:
        res_tag, filled, pnl = "UNFILLED", False, 0.0
        for idx_f, bar in df.iloc[i + 1 : i + 18].iterrows():
            if not filled and bar["Low"] <= trade["entry"] <= bar["High"]:
                filled, res_tag = True, "OPEN"
            if filled:
                hit_sl = bar["Low"] <= trade["sl"] if trade["side"] == "BUY" else bar["High"] >= trade["sl"]
                hit_tp = bar["High"] >= trade["tp"] if trade["side"] == "BUY" else bar["Low"] <= trade["tp"]
                if hit_sl:
                    res_tag, pnl = "LOSS", -1.0
                    break
                if hit_tp:
                    res_tag, pnl = "WIN", rr_target
                    break
        trade.update({"result": res_tag, "pnl": pnl})
        journal.append(trade)

wins = len([t for t in journal if t["result"] == "WIN"])
losses = len([t for t in journal if t["result"] == "LOSS"])
closed_n = wins + losses
win_rate = int((wins / closed_n) * 100) if closed_n else 0
net_r = sum(t["pnl"] for t in journal)

st.metric(f"Live {symbol} ({tf}) • Win Rate: {win_rate}% ({wins}W-{losses}L) • Net: {net_r:+.1f}R", f"${float(df['Close'].iloc[-1]):,.2f}")

if journal:
    last = journal[-1]
    msg = f"LATEST PULLBACK SETUP ({last['time']} EAT): **{last['side']} LIMIT** [{last['result']}] | Entry: **${last['entry']}** | SL: **${last['sl']}** | TP: **${last['tp']}**"
    st.success(msg) if last["side"] == "BUY" else st.error(msg)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="50 EMA", line=dict(color="#38bdf8", width=1.4)))
fig.add_trace(go.Scatter(x=df.index, y=df["ema100"], mode="lines", name="100 EMA", line=dict(color="#f59e0b", width=1.8)))

for side, col, sym, pos in [("BUY", "#00ff88", "triangle-up", "bottom center"), ("SELL", "#ff4b4b", "triangle-down", "top center")]:
    pts = [t for t in journal if t["side"] == side]
    if pts:
        fig.add_trace(go.Scatter(x=[p["ts"] for p in pts], y=[p["y"] for p in pts], mode="markers+text", name=side, text=[f"{side} ({p['result']})" for p in pts], textposition=pos, textfont=dict(color=col, size=10), marker=dict(symbol=sym, size=14, color=col)))

if journal:
    last = journal[-1]
    fig.add_hline(y=last["entry"], line_color="#38bdf8", annotation_text=f"Limit Entry ${last['entry']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

if journal:
    st.subheader("High-Accuracy Pullback Journal")
    j_df = pd.DataFrame([{"Time (EAT)": t["time"], "Setup": f"{t['side']} LIMIT", "Result": t["result"], "P&L": f"{t['pnl']:+.1f}R", "Limit Entry": f"${t['entry']}", "SL": f"${t['sl']}", "TP": f"${t['tp']}"} for t in reversed(journal)])
    st.dataframe(j_df, hide_index=True, use_container_width=True)
