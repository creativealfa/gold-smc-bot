
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold Institutional Sweep Bot", page_icon="🪙", layout="wide")
EAT = ZoneInfo("Africa/Nairobi")

st.title("Gold Institutional Liquidity Sweep Bot")
st.caption(f"Judas Swing + FVG Displacement + Auto-Journal • Nairobi Time: {datetime.now(EAT).strftime('%H:%M EAT')}")

try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

api_key = st.text_input("Twelve Data API Key", value=saved_key, type="password").strip()
symbol = st.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
tf = st.selectbox("Timeframe", ["5min", "15min"], index=0)

if st.button("Refresh Live Candles"):
    st.cache_data.clear()

if not api_key:
    st.info("Paste your Twelve Data API Key above (or save it in Streamlit Secrets) to start.")
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
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df, "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)

df, status = get_candles(api_key, symbol, tf)
if df.empty:
    st.error(f"Data error ({symbol}): {status}")
    st.stop()

# Trend Filter (50 EMA) & ATR Volatility
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill()

# Detect Institutional Liquidity Sweeps + Fair Value Gaps (FVG)
journal = []
for i in range(23, len(df)):
    pool = df.iloc[i - 22 : i - 2]
    pool_hi, pool_lo = float(pool["High"].max()), float(pool["Low"].min())
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    killzone = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)
    displaced = abs(c2["Close"] - c2["Open"]) > (c2["atr"] * 0.75)

    trade = None
    # Bearish Sweep + FVG
    if ((c1["High"] > pool_hi + 0.5) or (c2["High"] > pool_hi + 0.5)) and (c3["Close"] < pool_hi) and (c1["Low"] > c3["High"]):
        peak = max(c1["High"], c2["High"])
        score = 50 + (20 if killzone else 0) + (15 if displaced else 0) + (15 if c3["Close"] < c3["ema50"] else 0)
        entry = round((c1["Low"] + c3["High"]) / 2, 2)
        sl = round(peak + 0.80, 2)
        tp = round(entry - max(sl - entry, 0.50) * 2.5, 2)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "SELL", "y": float(c3["High"]) + 1.2, "entry": entry, "sl": sl, "tp": tp, "score": score}

    # Bullish Sweep + FVG
    elif ((c1["Low"] < pool_lo - 0.5) or (c2["Low"] < pool_lo - 0.5)) and (c3["Close"] > pool_lo) and (c1["High"] < c3["Low"]):
        bottom = min(c1["Low"], c2["Low"])
        score = 50 + (20 if killzone else 0) + (15 if displaced else 0) + (15 if c3["Close"] > c3["ema50"] else 0)
        entry = round((c1["High"] + c3["Low"]) / 2, 2)
        sl = round(bottom - 0.80, 2)
        tp = round(entry + max(entry - sl, 0.50) * 2.5, 2)
        trade = {"ts": ts, "time": ts.strftime("%m-%d %H:%M"), "side": "BUY", "y": float(c3["Low"]) - 1.2, "entry": entry, "sl": sl, "tp": tp, "score": score}

    if trade:
        res_tag, filled, pnl = "PENDING", False, 0.0
        for idx_f, bar in df.iloc[i + 1 :].iterrows():
            if not filled and bar["Low"] <= trade["entry"] <= bar["High"]:
                filled, res_tag = True, "OPEN"
            if filled:
                hit_sl = bar["Low"] <= trade["sl"] if trade["side"] == "BUY" else bar["High"] >= trade["sl"]
                hit_tp = bar["High"] >= trade["tp"] if trade["side"] == "BUY" else bar["Low"] <= trade["tp"]
                if hit_sl:
                    res_tag, pnl = "LOSS", -1.0
                    break
                if hit_tp:
                    res_tag, pnl = "WIN", 2.5
                    break
        trade.update({"result": res_tag, "pnl": pnl})
        journal.append(trade)

wins = len([t for t in journal if t["result"] == "WIN"])
losses = len([t for t in journal if t["result"] == "LOSS"])
net_r = sum(t["pnl"] for t in journal)
st.metric(f"Live {symbol} • Journal: {wins}W - {losses}L ({net_r:+.1f}R)", f"${float(df['Close'].iloc[-1]):,.2f}")

if journal:
    last = journal[-1]
    banner = f"LATEST SETUP ({last['time']} EAT): **{last['side']}** [{last['result']}] | Confidence: **{last['score']}%** | Entry: **${last['entry']}** | SL: **${last['sl']}** | TP: **${last['tp']}**"
    st.success(banner) if last["side"] == "BUY" else st.error(banner)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="50 EMA", line=dict(color="#f59e0b", width=1.5)))

for side, col, sym, pos in [("BUY", "#00ff88", "triangle-up", "bottom center"), ("SELL", "#ff4b4b", "triangle-down", "top center")]:
    pts = [t for t in journal if t["side"] == side]
    if pts:
        fig.add_trace(go.Scatter(x=[p["ts"] for p in pts], y=[p["y"] for p in pts], mode="markers+text", name=side, text=[f"{side} ({p['result']})" for p in pts], textposition=pos, textfont=dict(color=col, size=10), marker=dict(symbol=sym, size=14, color=col)))

if journal:
    last = journal[-1]
    fig.add_hline(y=last["entry"], line_color="#38bdf8", annotation_text=f"Entry ${last['entry']}")
    fig.add_hline(y=last["sl"], line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${last['sl']}")
    fig.add_hline(y=last["tp"], line_dash="dash", line_color="#00ff88", annotation_text=f"TP ${last['tp']}")

fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

if journal:
    st.subheader("Automated Trade Journal")
    j_df = pd.DataFrame([{"Time (EAT)": t["time"], "Side": t["side"], "Outcome": t["result"], "P&L": f"{t['pnl']:+.1f}R", "Conf": f"{t['score']}%", "Entry": f"${t['entry']}", "SL": f"${t['sl']}", "TP": f"${t['tp']}"} for t in reversed(journal)])
    st.dataframe(j_df, hide_index=True, use_container_width=True)
