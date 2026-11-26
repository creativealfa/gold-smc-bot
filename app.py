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
st.caption(f"Live Candles + Auto Win/Loss Journal + Mistake Adaptation • Nairobi Time: {datetime.now(EAT_TZ).strftime('%H:%M:%S EAT')}")

saved_key = ""
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
    st.warning("Paste your Twelve Data API Key above to activate live candles and the AI journal.")
    st.stop()

@st.cache_data(ttl=60)
def fetch_live_candles(key: str, symbol: str, interval: str):
    base_url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": 240, "timezone": "Africa/Nairobi", "apikey": key}
    try:
        resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(4, 7))
        res = resp.json()
        if "values" not in res:
            return pd.DataFrame(), res.get("message", str(res))
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

df, status_msg = fetch_live_candles(api_key, symbol_feed, timeframe)
if df.empty:
    st.error(f"Could not load live candles ({symbol_feed}): {status_msg}")
    st.stop()

# 1. Market Blueprint (Institutional Limit POC)
p_min, p_max = float(df["Low"].min()), float(df["High"].max())
bins = np.linspace(p_min, p_max, 28)
bin_centers = (bins[:-1] + bins[1:]) / 2
order_density = np.zeros(len(bin_centers))

for lo, hi, body in zip(df["Low"].to_numpy(), df["High"].to_numpy(), np.abs(df["Close"].to_numpy() - df["Open"].to_numpy())):
    rng = max(hi - lo, 0.10)
    wick_weight = 1.0 + ((rng - body) / rng) * 2.0
    order_density[(bin_centers >= lo) & (bin_centers <= hi)] += wick_weight

poc_price = round(float(bin_centers[np.argmax(order_density)]), 2)
high_density_levels = bin_centers[order_density >= np.percentile(order_density, 75)]

# 2. Multi-Model Indicators (CRT + ICT + SMC + Trend)
df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["trend_slope"] = df["ema20"].diff(5).fillna(0)
tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean().bfill()
df["body_atr"] = (df["Close"] - df["Open"]).abs() / df["atr"].replace(0, 1)
df["crt_range_ratio"] = (df["High"].rolling(6).max() - df["Low"].rolling(6).min()) / (df["atr"].replace(0, 1) * 6)
roll_hi, roll_lo = df["High"].rolling(24).max(), df["Low"].rolling(24).min()
df["ict_pd_level"] = ((df["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, 1e-9)).fillna(0.5)

# 3. Scan Setups & Grade Historical Outcomes for the Trading Journal
raw_trades = []
lookback, buf, rr = 20, 0.50, 2.0

for i in range(lookback + 3, len(df)):
    prior = df.iloc[i - lookback - 2 : i - 2]
    s_high, s_low = float(prior["High"].max()), float(prior["Low"].min())
    crt_block = df.iloc[i - 6 : i - 2]
    crt_hi, crt_lo = float(crt_block["High"].max()), float(crt_block["Low"].min())

    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    ts = df.index[i]
    in_killzone = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)

    setup = None
    if ((c1["High"] > s_high + buf) or (c2["High"] > s_high + buf)) and (c3["Close"] < s_high) and (c1["Low"] > c3["High"]):
        peak = max(c1["High"], c2["High"])
        cords = {}
        cords["[BLUEPRINT] Resting Buy-Stop Pool Swept"] = peak >= s_high + buf
        cords["[BLUEPRINT] Institutional Limit Node Reaction"] = any(abs(peak - lvl) <= 2.0 for lvl in high_density_levels)
        cords["[CRT] Range High Purge & Reclaim"] = (peak > crt_hi) and (c3["Close"] < crt_hi)
        cords["[CRT] Pre-Sweep AMD Compression"] = c1["crt_range_ratio"] < 0.85
        cords["[ICT] Premium Zone Entry (>0.62 Fib)"] = max(c1["ict_pd_level"], c2["ict_pd_level"]) >= 0.62
        cords["[ICT] Killzone Session Timing"] = in_killzone
        cords["[SMC] MSS + Displacement (>0.8x ATR)"] = (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8)
        cords["[SMC] Clean Bearish FVG + OB"] = ((c1["Low"] - c3["High"]) >= 0.20) and (c1["Close"] > c1["Open"])
        cords["[TREND] Trendline Breakdown / Slope"] = (c3["trend_slope"] < 0) or (c3["Close"] < c2["Low"])
        cords["[TREND] Bearish EMA 20/50 Alignment"] = c3["ema20"] < c3["ema50"]

        entry = round((c1["Low"] + c3["High"]) / 2, 2)
        sl = round(peak + 0.80, 2)
        tp = round(entry - max(sl - entry, 0.50) * rr, 2)
        setup = {"idx": i, "timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "SELL", "marker_y": float(c3["High"]) + 1.3, "entry": entry, "sl": sl, "tp": tp, "cords": cords}

    elif ((c1["Low"] < s_low - buf) or (c2["Low"] < s_low - buf)) and (c3["Close"] > s_low) and (c1["High"] < c3["Low"]):
        bottom = min(c1["Low"], c2["Low"])
        cords = {}
        cords["[BLUEPRINT] Resting Sell-Stop Pool Swept"] = bottom <= s_low - buf
        cords["[BLUEPRINT] Institutional Limit Node Reaction"] = any(abs(bottom - lvl) <= 2.0 for lvl in high_density_levels)
        cords["[CRT] Range Low Purge & Reclaim"] = (bottom < crt_lo) and (c3["Close"] > crt_lo)
        cords["[CRT] Pre-Sweep AMD Compression"] = c1["crt_range_ratio"] < 0.85
        cords["[ICT] Discount Zone Entry (<0.38 Fib)"] = min(c1["ict_pd_level"], c2["ict_pd_level"]) <= 0.38
        cords["[ICT] Killzone Session Timing"] = in_killzone
        cords["[SMC] MSS + Displacement (>0.8x ATR)"] = (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8)
        cords["[SMC] Clean Bullish FVG + OB"] = ((c3["Low"] - c1["High"]) >= 0.20) and (c1["Close"] < c1["Open"])
        cords["[TREND] Trendline Breakout / Slope"] = (c3["trend_slope"] > 0) or (c3["Close"] > c2["High"])
        cords["[TREND] Bullish EMA 20/50 Alignment"] = c3["ema20"] > c3["ema50"]

        entry = round((c1["High"] + c3["Low"]) / 2, 2)
        sl = round(bottom - 0.80, 2)
        tp = round(entry + max(entry - sl, 0.50) * rr, 2)
        setup = {"idx": i, "timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "BUY", "marker_y": float(c3["Low"]) - 1.3, "entry": entry, "sl": sl, "tp": tp, "cords": cords}

    if setup is not None:
        # Simulate forward candles to record actual WIN, LOSS, OPEN, or PENDING outcome
        future_bars = df.iloc[i + 1 :]
        outcome = "PENDING"
        filled = False
        pnl_r = 0.0
        lesson_note = "Waiting for limit fill"

        for _, fbar in future_bars.iterrows():
            if not filled:
                if fbar["Low"] <= setup["entry"] <= fbar["High"]:
                    filled = True
                    outcome = "OPEN"
                    lesson_note = "Trade active - monitoring TP/SL"
            if filled:
                if setup["side"] == "BUY":
                    if fbar["Low"] <= setup["sl"]:
                        outcome = "LOSS"
                        pnl_r = -1.0
                        missed = [k.split("] ")[1] for k, v in setup["cords"].items() if not v]
                        lesson_note = "Mistake: Missing " + (", ".join(missed[:2]) if missed else "momentum follow-through")
                        break
                    elif fbar["High"] >= setup["tp"]:
                        outcome = "WIN"
                        pnl_r = rr
                        lesson_note = "Success: Clean liquidity sweep + FVG expansion"
                        break
                else:
                    if fbar["High"] >= setup["sl"]:
                        outcome = "LOSS"
                        pnl_r = -1.0
                        missed = [k.split("] ")[1] for k, v in setup["cords"].items() if not v]
                        lesson_note = "Mistake: Missing " + (", ".join(missed[:2]) if missed else "momentum follow-through")
                        break
                    elif fbar["Low"] <= setup["tp"]:
                        outcome = "WIN"
                        pnl_r = rr
                        lesson_note = "Success: Clean liquidity sweep + FVG expansion"
                        break

        setup["outcome"] = outcome
        setup["pnl_r"] = pnl_r
        setup["lesson"] = lesson_note
        raw_trades.append(setup)

# 4. Self-Learning Weight Adjustment Engine (Learns from Journal Wins vs Losses)
cord_names = list(raw_trades[0]["cords"].keys()) if raw_trades else []
cord_weights = {c: 1.0 for c in cord_names}
closed_trades = [t for t in raw_trades if t["outcome"] in ("WIN", "LOSS")]
wins_list = [t for t in closed_trades if t["outcome"] == "WIN"]
losses_list = [t for t in closed_trades if t["outcome"] == "LOSS"]

top_mistake_cord = "None yet"
best_winning_cord = "None yet"

if len(closed_trades) >= 2 and cord_names:
    mistake_scores = {}
    win_scores = {}
    for c in cord_names:
        # How often was this cord MISSING during losses?

