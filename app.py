import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Gold 10-Cord AI Terminal", page_icon="🧠", layout="wide")
EAT_TZ = ZoneInfo("Africa/Nairobi")

now_eat = datetime.now(EAT_TZ)
st.title("Gold 10-Cord AI & Market Blueprint Terminal")
st.caption(f"Order-Book Heatmap + CRT + ICT + SMC + Trend | Nairobi Time: {now_eat.strftime('%H:%M EAT')}")

saved_key = ""
try:
    saved_key = str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
except Exception:
    saved_key = ""

exp = st.expander("Terminal Settings & Timeframe", expanded=(not bool(saved_key)))
c_key, c_feed, c_tf, c_conf = exp.columns(4)
api_key = c_key.text_input("Twelve Data API Key", value=saved_key, type="password", placeholder="Paste API key...").strip()
symbol_feed = c_feed.selectbox("Gold Feed", ["PAXG/USD", "XAU/USD"], index=0)
timeframe = c_tf.selectbox("Timeframe", ["5min", "1min", "15min"], index=0)
min_conf = c_conf.slider("Min AI Confidence %", 40, 85, 50)

if not api_key:
    st.warning("Paste your Twelve Data API Key in the settings box above to load the live Gold chart.")
    st.stop()

@st.cache_data(ttl=180)
def fetch_gold_series(key: str, symbol: str, interval: str) -> tuple[pd.DataFrame, str]:
    base_url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": 180, "timezone": "Africa/Nairobi", "apikey": key}
    try:
        resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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
        return pd.DataFrame(), f"Connection error: {e}"

df, status_msg = fetch_gold_series(api_key, symbol_feed, timeframe)

if df.empty:
    st.error(f"Could not load {symbol_feed}: {status_msg}")
    st.stop()

def build_market_blueprint(df: pd.DataFrame) -> dict:
    p_min, p_max = float(df["Low"].min()), float(df["High"].max())
    bins = np.linspace(p_min, p_max, 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    order_density = np.zeros(len(bin_centers))

    for lo, hi, body in zip(df["Low"].to_numpy(), df["High"].to_numpy(), np.abs(df["Close"].to_numpy() - df["Open"].to_numpy())):
        rng = max(hi - lo, 0.10)
        wick_weight = 1.0 + ((rng - body) / rng) * 2.0
        order_density[(bin_centers >= lo) & (bin_centers <= hi)] += wick_weight

    for idx, price in enumerate(bin_centers):
        if abs(price % 50) < 1.5 or abs(price % 25) < 1.0:
            order_density[idx] *= 1.35

    poc_price = round(float(bin_centers[np.argmax(order_density)]), 2)

    highs = df["High"].rolling(5, center=True).max().dropna()
    swing_highs = highs[highs == df["High"].reindex(highs.index)].unique()
    eq_highs = [round(float(max(swing_highs[i], swing_highs[j])), 2) for i in range(len(swing_highs)) for j in range(i + 1, len(swing_highs)) if abs(swing_highs[i] - swing_highs[j]) <= 1.80]

    lows = df["Low"].rolling(5, center=True).min().dropna()
    swing_lows = lows[lows == df["Low"].reindex(lows.index)].unique()
    eq_lows = [round(float(min(swing_lows[i], swing_lows[j])), 2) for i in range(len(swing_lows)) for j in range(i + 1, len(swing_lows)) if abs(swing_lows[i] - swing_lows[j]) <= 1.80]

    return {"levels": bin_centers, "density": order_density, "poc": poc_price, "buy_stops": sorted(list(set(eq_highs)))[-3:], "sell_stops": sorted(list(set(eq_lows)))[:3]}

def engineer_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["trend_slope"] = d["ema20"].diff(5).fillna(0)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - d["Close"].shift()).abs(), (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean().bfill()
    d["body_atr"] = (d["Close"] - d["Open"]).abs() / d["atr"].replace(0, 1)
    d["crt_range_ratio"] = (d["High"].rolling(6).max() - d["Low"].rolling(6).min()) / (d["atr"].replace(0, 1) * 6)
    roll_hi, roll_lo = d["High"].rolling(24).max(), d["Low"].rolling(24).min()
    d["ict_pd_level"] = ((d["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, 1e-9)).fillna(0.5)
    return d

def learn_historical_winrate(df: pd.DataFrame, side: str, in_killzone: bool, trend_aligned: bool) -> int:
    if len(df) < 30:
        return 68
    fut_hi = df["High"].rolling(8).max().shift(-8).iloc[:-8]
    fut_lo = df["Low"].rolling(8).min().shift(-8).iloc[:-8]
    base = df.iloc[:-8]
    wins = ((fut_hi - base["Close"]) > (base["atr"] * 1.1)).mean() if side == "BUY" else ((base["Close"] - fut_lo) > (base["atr"] * 1.1)).mean()
    prob = float(wins) * 100 + (12 if in_killzone else 0) + (10 if trend_aligned else 0)
    return int(np.clip(prob, 45, 92))

def analyze_10cord_setups(df: pd.DataFrame, bp: dict, min_conf_val: int) -> list[dict]:
    df = engineer_unified_features(df)
    signals = []
    high_density_levels = bp["levels"][bp["density"] >= np.percentile(bp["density"], 75)]
    lookback, buf, rr = 20, 0.50, 2.0

    for i in range(lookback + 3, len(df) - 1):
        prior = df.iloc[i - lookback - 2 : i - 2]
        s_high, s_low = float(prior["High"].max()), float(prior["Low"].min())
        crt_block = df.iloc[i - 6 : i - 2]
        crt_hi, crt_lo = float(crt_block["High"].max()), float(crt_block["Low"].min())

        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        ts = df.index[i]
        in_killzone = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)

        if ((c1["High"] > s_high + buf) or (c2["High"] > s_high + buf)) and (c3["Close"] < s_high) and (c1["Low"] > c3["High"]):
            peak = max(c1["High"], c2["High"])
            trend_ok = c3["ema20"] < c3["ema50"]
            cords = {}
            cords["[BLUEPRINT] Resting Buy-Stop Pool Swept"] = any(abs(peak - lvl) <= 2.5 for lvl in bp["buy_stops"]) or (peak >= s_high + buf)
            cords["[BLUEPRINT] Institutional Limit Node Reaction"] = any(abs(peak - lvl) <= 2.0 for lvl in high_density_levels)
            cords["[CRT] Range High Purge & Reclaim"] = (peak > crt_hi) and (c3["Close"] < crt_hi)
            cords["[CRT] Pre-Sweep AMD Compression"] = c1["crt_range_ratio"] < 0.85
            cords["[ICT] Premium Zone Entry (>0.62 Fib)"] = max(c1["ict_pd_level"], c2["ict_pd_level"]) >= 0.62
            cords["[ICT] Killzone Session Timing"] = in_killzone
            cords["[SMC] MSS + Displacement (>0.8x ATR)"] = (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8)
            cords["[SMC] Clean Bearish FVG + OB"] = ((c1["Low"] - c3["High"]) >= 0.20) and (c1["Close"] > c1["Open"])
            cords["[TREND] Trendline Breakdown / Slope"] = (c3["trend_slope"] < 0) or (c3["Close"] < c2["Low"])
            cords["[TREND] Bearish EMA 20/50 Alignment"] = trend_ok

            cord_score = sum(cords.values())
            ai_prob = learn_historical_winrate(df.iloc[:i], "SELL", in_killzone, trend_ok)
            final_conf = int((ai_prob * 0.40) + ((cord_score / 10) * 100 * 0.60))

            if final_conf >= min_conf_val:
                entry = round((c1["Low"] + c3["High"]) / 2, 2)
                sl = round(peak + 0.80, 2)
                tp = round(entry - max(sl - entry, 0.50) * rr, 2)
                badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
                signals.append({"timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "SELL", "marker_y": float(c3["High"]) + 1.3, "entry": entry, "sl": sl, "tp": tp, "confidence": final_conf, "ai_prob": ai_prob, "cords": cords, "badge": badge})

        if ((c1["Low"] < s_low - buf) or (c2["Low"] < s_low - buf)) and (c3["Close"] > s_low) and (c1["High"] < c3["Low"]):
            bottom = min(c1["Low"], c2["Low"])
            trend_ok = c3["ema20"] > c3["ema50"]
            cords = {}
            cords["[BLUEPRINT] Resting Sell-Stop Pool Swept"] = any(abs(bottom - lvl) <= 2.5 for lvl in bp["sell_stops"]) or (bottom <= s_low - buf)
            cords["[BLUEPRINT] Institutional Limit Node Reaction"] = any(abs(bottom - lvl) <= 2.0 for lvl in high_density_levels)
            cords["[CRT] Range Low Purge & Reclaim"] = (bottom < crt_lo) and (c3["Close"] > crt_lo)
            cords["[CRT] Pre-Sweep AMD Compression"] = c1["crt_range_ratio"] < 0.85
            cords["[ICT] Discount Zone Entry (<0.38 Fib)"] = min(c1["ict_pd_level"], c2["ict_pd_level"]) <= 0.38
            cords["[ICT] Killzone Session Timing"] = in_killzone
            cords["[SMC] MSS + Displacement (>0.8x ATR)"] = (c2["body_atr"] > 0.8) or (c3["body_atr"] > 0.8)
            cords["[SMC] Clean Bullish FVG + OB"] = ((c3["Low"] - c1["High"]) >= 0.20) and (c1["Close"] < c1["Open"])
            cords["[TREND] Trendline Breakout / Slope"] = (c3["trend_slope"] > 0) or (c3["Close"] > c2["High"])
            cords["[TREND] Bullish EMA 20/50 Alignment"] = trend_ok

            cord_score = sum(cords.values())
            ai_prob = learn_historical_winrate(df.iloc[:i], "BUY", in_killzone, trend_ok)
            final_conf = int((ai_prob * 0.40) + ((cord_score / 10) * 100 * 0.60))

            if final_conf >= min_conf_val:
                entry = round((c1["High"] + c3["Low"]) / 2, 2)
                sl = round(bottom - 0.80, 2)
