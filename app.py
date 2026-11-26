import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold Live AI Terminal", page_icon="📈", layout="wide")
EAT_TZ = ZoneInfo("Africa/Nairobi")

st.title("Gold Live AI & Market Blueprint Terminal")
st.caption(f"Live Candles + Buy/Sell Signals + 10-Cord Confluence • Nairobi Time: {datetime.now(EAT_TZ).strftime('%H:%M:%S EAT')}")

saved_key = ""
try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

api_key = st.text_input("Twelve Data API Key", value=saved_key, type="password").strip()
symbol_feed = st.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
timeframe = st.selectbox("Timeframe", ["5min", "1min", "15min"], index=0)
min_conf = st.slider("Min AI Confidence %", 40, 85, 50)

if not api_key:
    st.warning("Paste your Twelve Data API Key above to load the live candles.")
    st.stop()

base_url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
params = {"symbol": symbol_feed, "interval": timeframe, "outputsize": 120, "timezone": "Africa/Nairobi", "apikey": api_key}

try:
    resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(4, 6))
    res = resp.json()
except Exception as e:
    st.error(f"Network timeout or error: {e}")
    st.stop()

if "values" not in res:
    st.error(f"Market Data Notice ({symbol_feed}): {res.get('message', res)}")
    st.stop()

df = pd.DataFrame(res["values"])
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime").sort_index()
df.index = df.index.tz_localize(EAT_TZ)
for col in ["open", "high", "low", "close"]:
    df[col] = df[col].astype(float)
df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

# Market Blueprint (Point of Control & Resting Stop Pools)
p_min, p_max = float(df["Low"].min()), float(df["High"].max())
bins = np.linspace(p_min, p_max, 25)
bin_centers = (bins[:-1] + bins[1:]) / 2
order_density = np.zeros(len(bin_centers))

for lo, hi, body in zip(df["Low"].to_numpy(), df["High"].to_numpy(), np.abs(df["Close"].to_numpy() - df["Open"].to_numpy())):
    rng = max(hi - lo, 0.10)
    wick_weight = 1.0 + ((rng - body) / rng) * 2.0
    order_density[(bin_centers >= lo) & (bin_centers <= hi)] += wick_weight

poc_price = round(float(bin_centers[np.argmax(order_density)]), 2)
high_density_levels = bin_centers[order_density >= np.percentile(order_density, 75)]

# Indicators (CRT + ICT + SMC + Trend)
df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["trend_slope"] = df["ema20"].diff(5).fillna(0)
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill()
df["body_atr"] = (df["Close"] - df["Open"]).abs() / df["atr"].replace(0, 1)
df["crt_range_ratio"] = (df["High"].rolling(6).max() - df["Low"].rolling(6).min()) / (df["atr"].replace(0, 1) * 6)
roll_hi, roll_lo = df["High"].rolling(24).max(), df["Low"].rolling(24).min()
df["ict_pd_level"] = ((df["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, 1e-9)).fillna(0.5)

# 10-Cord Signal Scanner
signals = []
lookback, buf, rr = 20, 0.50, 2.0

for i in range(lookback + 3, len(df)):
    prior = df.iloc[i - lookback - 2 : i - 2]
    s_high, s_low = float(prior["High"].max()), float(prior["Low"].min())
    crt_block = df.iloc[i - 6 : i - 2]
    crt_hi, crt_lo = float(crt_block["High"].max()), float(crt_block["Low"].min())

    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    in_killzone = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)

    if ((c1["High"] > s_high + buf) or (c2["High"] > s_high + buf)) and (c3["Close"] < s_high) and (c1["Low"] > c3["High"]):
        peak = max(c1["High"], c2["High"])
        cords = {
            "[BLUEPRINT] Resting Buy-Stop Pool Swept": peak >= s_high + buf,
            "[BLUEPRINT] Institutional Limit Node Reaction": any(abs(peak - lvl) <= 2.0 for lvl in high_density_levels),
            "[CRT] Range High Purge & Reclaim": (peak > crt_hi) and (c3["Close"] < crt_hi),
            "[CRT] Pre-Sweep AMD Compression": c1["crt_range_ratio"] < 0.85,
            "[ICT] Premium Zone Entry (>0.62 Fib)": max(c1["ict_pd_level"], c2["ict_pd_level"]) >= 0.62,
            "[ICT] Killzone Session Timing": in_killzone,
            "[SMC] MSS + Displacement (>0.8x ATR)": (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8),
            "[SMC] Clean Bearish FVG + OB": ((c1["Low"] - c3["High"]) >= 0.20) and (c1["Close"] > c1["Open"]),
            "[TREND] Trendline Breakdown / Slope": (c3["trend_slope"] < 0) or (c3["Close"] < c2["Low"]),
            "[TREND] Bearish EMA 20/50 Alignment": c3["ema20"] < c3["ema50"],
        }
        final_conf = int((sum(cords.values()) / 10) * 100)
        if final_conf >= min_conf:
            entry = round((c1["Low"] + c3["High"]) / 2, 2)
            sl = round(peak + 0.80, 2)
            tp = round(entry - max(sl - entry, 0.50) * rr, 2)
            badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
            signals.append({"timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "SELL", "marker_y": float(c3["High"]) + 1.3, "entry": entry, "sl": sl, "tp": tp, "confidence": final_conf, "cords": cords, "badge": badge})

    if ((c1["Low"] < s_low - buf) or (c2["Low"] < s_low - buf)) and (c3["Close"] > s_low) and (c1["High"] < c3["Low"]):
        bottom = min(c1["Low"], c2["Low"])
        cords = {
            "[BLUEPRINT] Resting Sell-Stop Pool Swept": bottom <= s_low - buf,
            "[BLUEPRINT] Institutional Limit Node Reaction": any(abs(bottom - lvl) <= 2.0 for lvl in high_density_levels),
            "[CRT] Range Low Purge & Reclaim": (bottom < crt_lo) and (c3["Close"] > crt_lo),
            "[CRT] Pre-Sweep AMD Compression": c1["crt_range_ratio"] < 0.85,
            "[ICT] Discount Zone Entry (<0.38 Fib)": min(c1["ict_pd_level"], c2["ict_pd_level"]) <= 0.38,
            "[ICT] Killzone Session Timing": in_killzone,
            "[SMC] MSS + Displacement (>0.8x ATR)": (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8),
            "[SMC] Clean Bullish FVG + OB": ((c3["Low"] - c1["High"]) >= 0.20) and (c1["Close"] < c1["Open"]),
            "[TREND] Trendline Breakout / Slope": (c3["trend_slope"] > 0) or (c3["Close"] > c2["High"]),
            "[TREND] Bullish EMA 20/50 Alignment": c3["ema20"] > c3["ema50"],
        }
        final_conf = int((sum(cords.values()) / 10) * 100)
        if final_conf >= min_conf:
            entry = round((c1["High"] + c3["Low"]) / 2, 2)
            sl = round(bottom - 0.80, 2)
            tp = round(entry + max(entry - sl, 0.50) * rr, 2)
            badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
            signals.append({"timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "BUY", "marker_y": float(c3["Low"]) - 1.3, "entry": entry, "sl": sl, "tp": tp, "confidence": final_conf, "cords": cords, "badge": badge})

current_price = float(df["Close"].iloc[-1])
st.metric(f"Live {symbol_feed} Price (Blueprint Limit POC: ${poc_price:,.2f})", f"${current_price:,.2f}")

if signals:
    last = signals[-1]
    passed = sum(last["cords"].values())
    action_msg = f"LIVE AI SIGNAL ({last['time_str']} EAT): **{last['side']} GOLD** | Confidence: **{last['confidence']}% ({last['badge']})** | Cords: **{passed}/10** | Entry: **${last['entry']}** | SL: **${last['sl']}** | TP: **${last['tp']}**"
    if last["side"] == "BUY":
        st.success(action_msg)
    else:
        st.error(action_msg)

    st.subheader("10-Cord Confluence Badge Breakdown")
    for cord_name, ok in last["cords"].items():
        icon = "✅" if ok else "⚠️"
        st.markdown(f"{icon} **{cord_name}**")
else:
    st.info(f"LIVE SCANNER ACTIVE: Watching {len(df)} candles for a sweep + FVG setup >= **{min_conf}%** confidence.")

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol_feed))
fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], mode="lines", name="EMA 20", line=dict(color="#38bdf8", width=1.2)))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="EMA 50", line=dict(color="#f59e0b", width=1.5)))
fig.add_hline(y=poc_price, line_dash="dot", line_color="#a855f7", annotation_text=f"Blueprint Limit POC ${poc_price}")

buy_sigs = [s for s in signals if s["side"] == "BUY"]
sell_sigs = [s for s in signals if s["side"] == "SELL"]

if buy_sigs:
    fig.add_trace(go.Scatter(x=[s["timestamp"] for s in buy_sigs], y=[s["marker_y"] for s in buy_sigs], mode="markers+text", name="BUY", text=[f"BUY {s['confidence']}%" for s in buy_sigs], textposition="bottom center", textfont=dict(color="#00ff88", size=11), marker=dict(symbol="triangle-up", size=15, color="#00ff88", line=dict(width=1, color="black"))))

if sell_sigs:
    fig.add_trace(go.Scatter(x=[s["timestamp"] for s in sell_sigs], y=[s["marker_y"] for s in sell_sigs], mode="markers+text", name="SELL", text=[f"SELL {s['confidence']}%" for s in sell_sigs], textposition="top center", textfont=dict(color="#ff4b4b", size=11), marker=dict(symbol="triangle-down", size=15, color="#ff4b4b", line=dict(width=1, color="black"))))

if signals:
    last = signals[-1]
    fig.add_hline(y=last["entry"], line_dash="solid", line_color="#38bdf8", annotation_text=f"Entry ${last['entry']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
