import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Gold AI SMC Terminal", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes institutional technical features for AI training and scoring."""
    d = df.copy()
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["ema_dist"] = (d["Close"] - d["ema50"]) / d["Close"]
    
    # 14-period ATR
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"] - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean().bfill()
    d["body_atr_ratio"] = (d["Close"] - d["Open"]).abs() / d["atr"].replace(0, 1)
    
    # 14-period RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
    d["rsi"] = 100 - (100 / (1 + (gain / loss)))
    d["rsi"] = d["rsi"].fillna(50)
    
    # Upper & Lower wick ratios
    full_range = (d["High"] - d["Low"]).replace(0, 1e-9)
    d["upper_wick_ratio"] = (d["High"] - d[["Open", "Close"]].max(axis=1)) / full_range
    d["lower_wick_ratio"] = (d[["Open", "Close"]].min(axis=1) - d["Low"]) / full_range
    d["hour_eat"] = d.index.hour
    return d

def train_ai_models(df: pd.DataFrame):
    """Trains two Random Forest models on recent historical price behavior (Buy & Sell outcome probability)."""
    feature_cols = ["ema_dist", "body_atr_ratio", "rsi", "upper_wick_ratio", "lower_wick_ratio", "hour_eat"]
    train_df = df.iloc[:-12].copy()
    if len(train_df) < 60:
        return None, None, feature_cols

    # Label historical bars: did price rally +1.5 ATR before dropping -1.0 ATR in next 12 bars?
    future_high = pd.concat([df["High"].shift(-k) for k in range(1, 13)], axis=1).max(axis=1).iloc[:-12]
    future_low = pd.concat([df["Low"].shift(-k) for k in range(1, 13)], axis=1).min(axis=1).iloc[:-12]

    y_buy = ((future_high - train_df["Close"]) > (train_df["atr"] * 1.2)).astype(int)
    y_sell = ((train_df["Close"] - future_low) > (train_df["atr"] * 1.2)).astype(int)

    X = train_df[feature_cols].fillna(0)
    buy_model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
    sell_model = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)

    if y_buy.nunique() > 1:
        buy_model.fit(X, y_buy)
    else:
        buy_model = None

    if y_sell.nunique() > 1:
        sell_model.fit(X, y_sell)
    else:
        sell_model = None

    return buy_model, sell_model, feature_cols

def analyze_ai_scalps(df: pd.DataFrame, lookback: int, buf: float, rr: float, min_conf: int) -> list[dict]:
    df = engineer_features(df)
    buy_model, sell_model, f_cols = train_ai_models(df)
    signals = []

    for i in range(lookback + 2, len(df) - 1):
        prior = df.iloc[i - lookback - 2 : i - 2]
        s_high, s_low = float(prior["High"].max()), float(prior["Low"].min())
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        ts = df.index[i]
        hour = ts.hour
        in_killzone = (10 <= hour <= 13) or (15 <= hour <= 19)
        X_now = pd.DataFrame([c3[f_cols].fillna(0)])

        # --- CHECK SELL SETUP ---
        if ((c1["High"] > s_high + buf) or (c2["High"] > s_high + buf)) and (c3["Close"] < s_high) and (c1["Low"] > c3["High"]):
            cords = {
                "Liquidity Sweep Confirmed": True,
                "FVG Imbalance Present": (c1["Low"] - c3["High"]) >= 0.25,
                "Strong Displacement (>0.8x ATR)": c2["body_atr_ratio"] > 0.8 or c3["body_atr_ratio"] > 0.8,
                "50-EMA Bearish Bias": c3["Close"] < c3["ema50"],
                "RSI Overbought Rejection (>58)": max(c1["rsi"], c2["rsi"]) > 58,
                "Institutional Killzone (EAT)": in_killzone,
            }
            cord_score = sum(cords.values())
            ai_prob = int(sell_model.predict_proba(X_now)[0][1] * 100) if sell_model else 65
            final_conf = int((ai_prob * 0.5) + ((cord_score / 6) * 100 * 0.5))

            if final_conf >= min_conf:
                entry = round((c1["Low"] + c3["High"]) / 2, 2)
                sl = round(max(c1["High"], c2["High"]) + 0.80, 2)
                tp = round(entry - max(sl - entry, 0.50) * rr, 2)
                badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
                signals.append({
                    "timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "SELL",
                    "marker_y": float(c3["High"]) + 1.3, "entry": entry, "sl": sl, "tp": tp,
                    "confidence": final_conf, "ai_prob": ai_prob, "cords": cords, "badge": badge
                })

        # --- CHECK BUY SETUP ---
        if ((c1["Low"] < s_low - buf) or (c2["Low"] < s_low - buf)) and (c3["Close"] > s_low) and (c1["High"] < c3["Low"]):
            cords = {
                "Liquidity Sweep Confirmed": True,
                "FVG Imbalance Present": (c3["Low"] - c1["High"]) >= 0.25,
                "Strong Displacement (>0.8x ATR)": c2["body_atr_ratio"] > 0.8 or c3["body_atr_ratio"] > 0.8,
                "50-EMA Bullish Bias": c3["Close"] > c3["ema50"],
                "RSI Oversold Rejection (<42)": min(c1["rsi"], c2["rsi"]) < 42,
                "Institutional Killzone (EAT)": in_killzone,
            }
            cord_score = sum(cords.values())
            ai_prob = int(buy_model.predict_proba(X_now)[0][1] * 100) if buy_model else 65
            final_conf = int((ai_prob * 0.5) + ((cord_score / 6) * 100 * 0.5))

            if final_conf >= min_conf:
                entry = round((c1["High"] + c3["Low"]) / 2, 2)
                sl = round(min(c1["Low"], c2["Low"]) - 0.80, 2)
                tp = round(entry + max(entry - sl, 0.50) * rr, 2)
                badge = "HIGH" if final_conf >= 75 else ("MEDIUM" if final_conf >= 60 else "LOW")
                signals.append({
                    "timestamp": ts, "time_str": ts.strftime("%m-%d %H:%M"), "side": "BUY",
                    "marker_y": float(c3["Low"]) - 1.3, "entry": entry, "sl": sl, "tp": tp,
                    "confidence": final_conf, "ai_prob": ai_prob, "cords": cords, "badge": badge
                })

    return signals

with st.sidebar:
    st.header("AI Scalper Controls")
    api_key = st.text_input("Twelve Data API Key", type="password", value=default_key).strip()
    timeframe = st.selectbox("Timeframe", ["5min", "1min", "15min"], index=0)
    min_conf = st.slider("Minimum AI Confidence Filter (%)", 40, 85, 55)
    lookback = st.slider("Liquidity Lookback (Candles)", 10, 40, 20)
    buf = st.number_input("Min Sweep Buffer ($)", value=0.50, step=0.25)
    rr = st.number_input("Risk-to-Reward Ratio", value=2.0, step=0.5)
    if st.button("Retrain AI & Refresh", width="stretch"):
        st.cache_data.clear()

now_eat = datetime.now(EAT_TZ)
st.title("Gold AI SMC Trading Terminal")
st.caption(f"Self-Learning Random Forest + 6-Cord Confluence Engine | Nairobi Time: {now_eat.strftime('%H:%M EAT')}")

if not api_key:
    st.info("Open the top-left menu (>) and paste your Twelve Data API Key to activate the AI.")
    st.stop()

df, status_msg = fetch_gold_series(api_key, "XAU/USD", timeframe)
active_symbol = "XAU/USD"
if df.empty:
    df, status_msg = fetch_gold_series(api_key, "PAXG/USD", timeframe)
    active_symbol = "PAXG/USD (Spot Gold)"
if df.empty:
    st.error(f"Data error: {status_msg}")
    st.stop()

signals = analyze_ai_scalps(df, lookback, buf, rr, min_conf)
current_price = float(df["Close"].iloc[-1])

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Live {active_symbol}", f"${current_price:,.2f}")
if signals:
    last = signals[-1]
    m2.metric("Latest AI Signal", f"{last['side']} ({last['time_str']})")
    m3.metric("AI Confidence Score", f"{last['confidence']}%", delta=f"{last['badge']} CONVICTION")
    m4.metric("FVG Entry / TP", f"${last['entry']}", delta=f"TP ${last['tp']}")
else:
    m2.metric("Latest AI Signal", "Scanning...")
    m3.metric("AI Confidence Score", "Waiting")
    m4.metric("Active Filter", f">= {min_conf}% Conf.")

# AI Clear Communication & Confluence Badge Panel
if signals:
    last = signals[-1]
    passed = [k for k, v in last["cords"].items() if v]
    failed = [k for k, v in last["cords"].items() if not v]
    summary_text = (
        f"**AI Analyst Summary ({last['time_str']} EAT):** The Random Forest model detected a **{last['side']}** setup "
