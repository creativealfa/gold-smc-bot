import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold Self-Learning AI Journal", page_icon="🧠", layout="wide")
EAT_TZ = ZoneInfo("Africa/Nairobi")
st.title("Gold Self-Learning AI & Trading Journal")
st.caption(f"Live Candles + Win/Loss Journal + Mistake Adaptation • Nairobi Time: {datetime.now(EAT_TZ).strftime('%H:%M:%S EAT')}")

try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

api_key = st.text_input("Twelve Data API Key", value=saved_key, type="password").strip()
symbol_feed = st.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
timeframe = st.selectbox("Timeframe", ["5min", "1min", "15min"], index=0)
min_conf = st.slider("Min Adjusted AI Confidence %", 40, 85, 50)

if st.button("Refresh Live Candles & Retrain Journal"):
    st.cache_data.clear()

if not api_key:
    st.warning("Paste your Twelve Data API Key above to activate live candles.")
    st.stop()

@st.cache_data(ttl=60)
def fetch_candles(key: str, symbol: str, interval: str):
    url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": 220, "timezone": "Africa/Nairobi", "apikey": key}
    try:
        res = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(4, 7)).json()
        if "values" not in res:
            return pd.DataFrame(), res.get("message", str(res))
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_localize(EAT_TZ)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)

df, msg = fetch_candles(api_key, symbol_feed, timeframe)
if df.empty:
    st.error(f"Market Data Notice ({symbol_feed}): {msg}")
    st.stop()

bins = np.linspace(float(df["Low"].min()), float(df["High"].max()), 26)
centers = (bins[:-1] + bins[1:]) / 2
density = np.zeros(len(centers))
for lo, hi, body in zip(df["Low"].to_numpy(), df["High"].to_numpy(), np.abs(df["Close"].to_numpy() - df["Open"].to_numpy())):
    rng = max(hi - lo, 0.10)
    density[(centers >= lo) & (centers <= hi)] += 1.0 + ((rng - body) / rng) * 2.0
poc = round(float(centers[np.argmax(density)]), 2)
hi_nodes = centers[density >= np.percentile(density, 75)]

df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["slope"] = df["ema20"].diff(5).fillna(0)
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill()
df["body_atr"] = (df["Close"] - df["Open"]).abs() / df["atr"].replace(0, 1)
df["crt_ratio"] = (df["High"].rolling(6).max() - df["Low"].rolling(6).min()) / (df["atr"].replace(0, 1) * 6)
r_hi, r_lo = df["High"].rolling(24).max(), df["Low"].rolling(24).min()
df["ict_pd"] = ((df["Close"] - r_lo) / (r_hi - r_lo).replace(0, 1e-9)).fillna(0.5)

trades = []
for i in range(23, len(df)):
    prior = df.iloc[i - 22 : i - 2]
    s_hi, s_lo = float(prior["High"].max()), float(prior["Low"].min())
    crt_hi, crt_lo = float(df["High"].iloc[i - 6 : i - 2].max()), float(df["Low"].iloc[i - 6 : i - 2].min())
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    kz = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)
    setup = None

    if ((c1["High"] > s_hi + 0.5) or (c2["High"] > s_hi + 0.5)) and (c3["Close"] < s_hi) and (c1["Low"] > c3["High"]):
        pk = max(c1["High"], c2["High"])
        cords = {"[BP] Stop-Pool": pk >= s_hi + 0.5, "[BP] Limit-Node": any(abs(pk - l) <= 2.0 for l in hi_nodes), "[CRT] Range-Purge": pk > crt_hi and c3["Close"] < crt_hi, "[CRT] AMD-Compress": c1["crt_ratio"] < 0.85, "[ICT] Premium>0.62": max(c1["ict_pd"], c2["ict_pd"]) >= 0.62, "[ICT] Killzone": kz, "[SMC] Displacement": c2["body_atr"] > 0.8 or c3["body_atr"] > 0.8, "[SMC] Bear-FVG+OB": (c1["Low"] - c3["High"]) >= 0.2 and c1["Close"] > c1["Open"], "[TREND] Slope-Down": c3["slope"] < 0 or c3["Close"] < c2["Low"], "[TREND] EMA20<50": c3["ema20"] < c3["ema50"]}
        en = round((c1["Low"] + c3["High"]) / 2, 2)
        sl = round(pk + 0.8, 2)
        tp = round(en - max(sl - en, 0.5) * 2.0, 2)
        setup = {"ts": ts, "t_str": ts.strftime("%m-%d %H:%M"), "side": "SELL", "my": float(c3["High"]) + 1.3, "en": en, "sl": sl, "tp": tp, "cords": cords}
    elif ((c1["Low"] < s_lo - 0.5) or (c2["Low"] < s_lo - 0.5)) and (c3["Close"] > s_lo) and (c1["High"] < c3["Low"]):
        bt = min(c1["Low"], c2["Low"])
        cords = {"[BP] Stop-Pool": bt <= s_lo - 0.5, "[BP] Limit-Node": any(abs(bt - l) <= 2.0 for l in hi_nodes), "[CRT] Range-Purge": bt < crt_lo and c3["Close"] > crt_lo, "[CRT] AMD-Compress": c1["crt_ratio"] < 0.85, "[ICT] Discount<0.38": min(c1["ict_pd"], c2["ict_pd"]) <= 0.38, "[ICT] Killzone": kz, "[SMC] Displacement": c2["body_atr"] > 0.8 or c3["body_atr"] > 0.8, "[SMC] Bull-FVG+OB": (c3["Low"] - c1["High"]) >= 0.2 and c1["Close"] < c1["Open"], "[TREND] Slope-Up": c3["slope"] > 0 or c3["Close"] > c2["High"], "[TREND] EMA20>50": c3["ema20"] > c3["ema50"]}
        en = round((c1["High"] + c3["Low"]) / 2, 2)
        sl = round(bt - 0.8, 2)
        tp = round(en + max(en - sl, 0.5) * 2.0, 2)
        setup = {"ts": ts, "t_str": ts.strftime("%m-%d %H:%M"), "side": "BUY", "my": float(c3["Low"]) - 1.3, "en": en, "sl": sl, "tp": tp, "cords": cords}

    if setup:
        out, filled, pnl, note = "PENDING", False, 0.0, "Waiting for FVG fill"
        for idx_f, fb in df.iloc[i + 1 :].iterrows():
            if not filled and fb["Low"] <= setup["en"] <= fb["High"]:
                filled, out, note = True, "OPEN", "Active trade"
            if filled:
                hit_sl = fb["Low"] <= setup["sl"] if setup["side"] == "BUY" else fb["High"] >= setup["sl"]
                hit_tp = fb["High"] >= setup["tp"] if setup["side"] == "BUY" else fb["Low"] <= setup["tp"]
                if hit_sl:
                    missed = [k for k, v in setup["cords"].items() if not v]
                    out, pnl, note = "LOSS", -1.0, "Mistake: Missing " + (", ".join(missed[:2]) if missed else "momentum")
                    break
                if hit_tp:
                    out, pnl, note = "WIN", 2.0, "Success: Clean sweep + FVG expansion"
                    break
        setup.update({"out": out, "pnl": pnl, "note": note})
        trades.append(setup)

c_names = list(trades[0]["cords"].keys()) if trades else []
closed = [t for t in trades if t["out"] in ("WIN", "LOSS")]
wins = [t for t in closed if t["out"] == "WIN"]
losses = [t for t in closed if t["out"] == "LOSS"]
m_scores = {c: (float(np.mean([not t["cords"][c] for t in losses])) if losses else 0.0) for c in c_names}
w_scores = {c: (float(np.mean([t["cords"][c] for t in wins])) if wins else 0.5) for c in c_names}
weights = {c: (round(float(np.clip(0.8 + w_scores[c] + m_scores[c] * 0.5, 0.5, 1.8)), 2) if len(closed) >= 2 else 1.0) for c in c_names}
top_err = max(m_scores, key=m_scores.get) if len(closed) >= 2 and m_scores else "None yet"

max_w = sum(weights.values()) if weights else 10.0
signals = []
for t in trades:
    t["conf"] = int((sum(weights[c] for c, ok in t["cords"].items() if ok) / max_w) * 100)
    if t["conf"] >= min_conf:
        signals.append(t)

wr = int((len(wins) / len(closed)) * 100) if closed else 0
st.metric(f"Live {symbol_feed} (POC: ${poc:,.2f}) | Journal: {len(wins)}W-{len(losses)}L ({wr}% WR)", f"${float(df['Close'].iloc[-1]):,.2f}")

if len(closed) >= 2 and top_err != "None yet":
    st.info(f"**AI Mistake Reflection:** Graded **{len(closed)}** closed trades ({sum(t['pnl'] for t in closed):+.1f}R). Most losses missed **{top_err}**—AI boosted its weight to **{weights[top_err]}x**.")
else:
    st.info("AI Journal active: Monitoring live candles to record wins, losses, and adjust cord weights.")

if signals:
    last = signals[-1]
    msg_txt = f"LATEST ({last['t_str']} EAT): **{last['side']}** [{last['out']}] | Adj. Conf: **{last['conf']}%** | Entry: **${last['en']}** | SL: **${last['sl']}** | TP: **${last['tp']}**"
    st.success(msg_txt) if last["side"] == "BUY" else st.error(msg_txt)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol_feed))
fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], mode="lines", name="EMA20", line=dict(color="#38bdf8", width=1.2)))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="EMA50", line=dict(color="#f59e0b", width=1.5)))
fig.add_hline(y=poc, line_dash="dot", line_color="#a855f7", annotation_text=f"POC ${poc}")

for side, col, sym, pos in [("BUY", "#00ff88", "triangle-up", "bottom center"), ("SELL", "#ff4b4b", "triangle-down", "top center")]:
    sub = [s for s in signals if s["side"] == side]
    if sub:
        fig.add_trace(go.Scatter(x=[s["ts"] for s in sub], y=[s["my"] for s in sub], mode="markers+text", name=side, text=[f"{side} ({s['out']})" for s in sub], textposition=pos, textfont=dict(color=col, size=10), marker=dict(symbol=sym, size=14, color=col)))

if signals:
    last = signals[-1]
    fig.add_hline(y=last["en"], line_color="#38bdf8", annotation_text=f"Entry ${last['en']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

