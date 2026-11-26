import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Gold AI & Market Blueprint Terminal", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
EAT_TZ = ZoneInfo("Africa/Nairobi")
default_key = st.secrets.get("TWELVE_DATA_API_KEY", "").strip() if hasattr(st, "secrets") else ""

@st.cache_data(ttl=60)
def fetch_gold_series(api_key: str, symbol: str, interval: str) -> tuple[pd.DataFrame, str]:
    if not api_key:
        return pd.DataFrame(), "No API key provided."
    base_url = "https://" + "api." + "twelvedata" + ".com" + "/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": 300, "timezone": "Africa/Nairobi", "apikey": api_key}
    try:
        resp = requests.get(base_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res = resp.json()
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

def build_market_blueprint(df: pd.DataFrame) -> dict:
    p_min, p_max = float(df["Low"].min()), float(df["High"].max())
    bins = np.linspace(p_min, p_max, 36)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    order_density = np.zeros(len(bin_centers))

    for _, row in df.iterrows():
        rng = max(row["High"] - row["Low"], 0.10)
        wick_weight = 1.0 + ((rng - abs(row["Close"] - row["Open"])) / rng) * 2.0
        mask = (bin_centers >= row["Low"]) & (bin_centers <= row["High"])
        order_density[mask] += wick_weight

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
    d["ema_spread"] = (d["ema20"] - d["ema50"]) / d["Close"]
    d["trend_slope"] = d["Close"].rolling(15).apply(lambda x: np.polyfit(range(len(x)), x, 1), raw=True).fillna(0)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - d["Close"].shift()).abs(), (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean().bfill()
    d["body_atr"] = (d["Close"] - d["Open"]).abs() / d["atr"].replace(0, 1)
    d["crt_range_ratio"] = (d["High"].rolling(6).max() - d["Low"].rolling(6).min()) / (d["atr"].replace(0, 1) * 6)
    roll_hi, roll_lo = d["High"].rolling(24).max(), d["Low"].rolling(24).min()
    d["ict_pd_level"] = ((d["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, 1e-9)).fillna(0.5)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
    d["rsi"] = (100 - (100 / (1 + (gain / loss)))).fillna(50)
    full_range = (d["High"] - d["Low"]).replace(0, 1e-9)
    d["upper_wick"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / full_range
    d["lower_wick"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / full_range
    d["hour_eat"] = d.index.hour
    return d

def train_unified_ai(df: pd.DataFrame):
    f_cols = ["ema_spread", "trend_slope", "body_atr", "crt_range_ratio", "ict_pd_level", "rsi", "upper_wick", "lower_wick", "hour_eat"]
    train_df = df.iloc[:-12].copy()
    if len(train_df) < 60:
        return None, None, f_cols
    fut_hi = pd.concat([df["High"].shift(-k) for k in range(1, 13)], axis=1).max(axis=1).iloc[:-12]
    fut_lo = pd.concat([df["Low"].shift(-k) for k in range(1, 13)], axis=1).min(axis=1).iloc[:-12]
    y_buy = ((fut_hi - train_df["Close"]) > (train_df["atr"] * 1.2)).astype(int)
    y_sell = ((train_df["Close"] - fut_lo) > (train_df["atr"] * 1.2)).astype(int)
    X = train_df[f_cols].fillna(0)
    b_mod = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    s_mod = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    b_mod = b_mod.fit(X, y_buy) if y_buy.nunique() > 1 else None
    s_mod = s_mod.fit(X, y_sell) if y_sell.nunique() > 1 else None
    return b_mod, s_mod, f_cols

def analyze_10cord_setups(df: pd.DataFrame, bp: dict, lookback: int, buf: float, rr: float, min_conf: int) -> list[dict]:
    df = engineer_unified_features(df)
    b_mod, s_mod, f_cols = train_unified_ai(df)
    signals = []
    high_density_levels = bp["levels"][bp["density"] >= np.percentile(bp["density"], 75)]

    for i in range(lookback + 3, len(df) - 1):
        prior = df.iloc[i - lookback - 2 : i - 2]
        s_high, s_low = float(prior["High"].max()), float(prior["Low"].min())
        crt_block = df.iloc[i - 6 : i - 2]
        crt_hi, crt_lo = float(crt_block["High"].max()), float(crt_block["Low"].min())

        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        ts = df.index[i]
        in_killzone = (10 <= ts.hour <= 13) or (15 <= ts.hour <= 19)
        X_now = pd.DataFrame([c3[f_cols].fillna(0)])

        # SELL CONFLUENCE (10 CORDS)
        if ((c1["High"] > s_high + buf) or (c2["High"] > s_high + buf)) and (c3["Close"] < s_high) and (c1["Low"] > c3["High"]):
            peak = max(c1["High"], c2["High"])
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
            cords["[TREND] Bearish EMA 20/50 Alignment"] = c3["ema20"] < c3["ema50"]

            cord_score = sum(cords.values())
            ai_prob = int(s_mod.predict_proba(X_now) * 100) if s_mod else 65
            final_conf = int((ai_prob * 0.40) + ((cord_score / 10) * 100 * 0.60))

            if final_conf >= min_conf:
                entry = round((c1["Low"] + c3["High"]) / 2, 2)
                sl = round(peak + 0.80, 2)
                tp = round(entry - max(sl - entry, 0.50) * rr, 2)
                badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
                signals.append({"timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "SELL", "marker_y": float(c3["High"]) + 1.3, "entry": entry, "sl": sl, "tp": tp, "confidence": final_conf, "ai_prob": ai_prob, "cords": cords, "badge": badge})

        # BUY CONFLUENCE (10 CORDS)
        if ((c1["Low"] < s_low - buf) or (c2["Low"] < s_low - buf)) and (c3["Close"] > s_low) and (c1["High"] < c3["Low"]):
            bottom = min(c1["Low"], c2["Low"])
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
            cords["[TREND] Bullish EMA 20/50 Alignment"] = c3["ema20"] > c3["ema50"]

            cord_score = sum(cords.values())
            ai_prob = int(b_mod.predict_proba(X_now) * 100) if b_mod else 65

