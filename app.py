import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="XAU/USD SMC Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
EAT_TZ = ZoneInfo("Africa/Nairobi")

@st.cache_data(ttl=60)
def fetch_live_gold(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "YOUR_TWELVE_DATA_API_KEY":
        return pd.DataFrame()
    url = f"https://twelvedata.com{api_key}"
    try:
        res = requests.get(url, timeout=8).json()
        if "values" not in res:
            return pd.DataFrame()
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_localize(EAT_TZ)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

with st.sidebar:
    st.header("Bot Settings")
    api_key = st.text_input("Twelve Data API Key", type="password", value="YOUR_TWELVE_DATA_API_KEY")
    if st.button("Refresh Data Now", width="stretch"):
        st.cache_data.clear()

now_eat = datetime.now(EAT_TZ)
st.title("Gold (XAU/USD) Liquidity Hunter")
st.caption(f"Live Session Tracker | Nairobi Time: {now_eat.strftime('%H:%M EAT')}")

df = fetch_live_gold(api_key)
if df.empty:
    st.info("Open the top-left menu (>) and paste your Twelve Data API Key to activate the live chart.")
    st.stop()

latest_date = df.index[-1].date()
london_df = df[df.index.date == latest_date].between_time("10:00", "14:59")
session_tag = "Today London"
if london_df.empty and len(df.index.normalize().unique()) > 1:
    prev_date = df.index.normalize().unique()[-2].date()
    london_df = df[df.index.date == prev_date].between_time("10:00", "14:59")
    session_tag = f"Prev London ({prev_date})"

current_price = df["Close"].iloc[-1]
london_high = float(london_df["High"].max()) if not london_df.empty else current_price
london_low = float(london_df["Low"].min()) if not london_df.empty else current_price

c1, c2, c3 = st.columns(3)
c1.metric("Live XAU/USD", f"${current_price:,.2f}")
c2.metric(f"{session_tag} High", f"${london_high:,.2f}", delta=f"${current_price - london_high:,.2f}")
c3.metric(f"{session_tag} Low", f"${london_low:,.2f}", delta=f"${current_price - london_low:,.2f}")

fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="XAU/USD")])
fig.add_hline(y=london_high, line_dash="dash", line_color="#ff4b4b", annotation_text="London High (BSL)")
fig.add_hline(y=london_low, line_dash="dash", line_color="#00cc96", annotation_text="London Low (SSL)")
fig.update_layout(template="plotly_dark", height=460, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")
