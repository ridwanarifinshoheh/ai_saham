# =========================================================
# AI SAHAM INDONESIA ULTRA PRO MAX
# FINAL POWERFUL VERSION
# REALTIME + MOBILE + TELEGRAM BOT
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from datetime import datetime
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=5000,
    key="refresh"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Saham Indonesia MAX",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# DARK MODE + MOBILE RESPONSIVE
# =========================================================

st.markdown(
    """
    <style>

    .stApp{
        background-color:#0d1117;
        color:white;
    }

    div[data-testid="metric-container"]{
        background:#161b22;
        border:1px solid #30363d;
        padding:15px;
        border-radius:15px;
    }

    @media (max-width: 768px){

        .block-container{
            padding-top:1rem;
            padding-left:1rem;
            padding-right:1rem;
        }

        h1{
            font-size:28px !important;
        }

        h2{
            font-size:22px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 AI Saham Indonesia MAX")
st.caption("Realtime Smart Money AI Dashboard")

# =========================================================
# MARKET STATUS
# =========================================================

current_hour = datetime.now().hour

if 9 <= current_hour <= 16:

    st.success("🟢 Market Open")

else:

    st.warning("🔴 Market Closed")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "1d"
    ],
    index=4
)

auto_telegram = st.sidebar.toggle(
    "Telegram Signal Bot",
    value=False
)

# =========================================================
# INPUT
# =========================================================

stock = st.text_input(
    "Kode Saham Utama",
    "BBCA"
)

compare_stocks = st.text_input(
    "Compare Saham",
    "BBRI,TLKM"
)

stock_code = stock.upper() + ".JK"

compare_list = [
    s.strip().upper() + ".JK"
    for s in compare_stocks.split(",")
]

# =========================================================
# TELEGRAM CONFIG
# =========================================================

TELEGRAM_TOKEN = "ISI_TOKEN_BOT"
TELEGRAM_CHAT_ID = "ISI_CHAT_ID"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(ttl=10)
def load_data(code, interval):

    # ==========================================
    # PERIOD SESUAI TIMEFRAME
    # ==========================================

    if interval == "1m":

        period = "7d"

    elif interval == "5m":

        period = "30d"

    elif interval == "15m":

        period = "60d"

    elif interval == "30m":

        period = "60d"

    elif interval == "1h":

        period = "90d"

    else:

        period = "1y"

    # ==========================================
    # DOWNLOAD DATA
    # ==========================================

    data = yf.download(
        code,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return data

    # ==========================================
    # FIX MULTI INDEX
    # ==========================================

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    return data

# =========================================================
# TELEGRAM FUNCTION
# =========================================================

def send_telegram(message):

    try:

        url = (
            f"https://api.telegram.org/bot"
            f"{TELEGRAM_TOKEN}/sendMessage"
        )

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }

        response = requests.post(
            url,
            data=payload,
            timeout=10
        )

        return response.status_code == 200

    except:

        return False

# =========================================================
# MAIN
# =========================================================

try:

    # =========================================================
    # LOAD DATA
    # =========================================================

    df = load_data(
        stock_code,
        timeframe
    )

    if df.empty:

        st.error("Data saham tidak ditemukan")
        st.stop()

    # =========================================================
    # CLEAN DATA
    # =========================================================

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    # =========================================================
    # MOVING AVERAGE
    # =========================================================

    df["MA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # =========================================================
    # RSI
    # =========================================================

    delta = df["Close"].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # =========================================================
    # SUPPORT RESISTANCE
    # =========================================================

    df["Support"] = (
        df["Low"]
        .rolling(20)
        .min()
    )

    df["Resistance"] = (
        df["High"]
        .rolling(20)
        .max()
    )

    # =========================================================
    # TRENDLINE
    # =========================================================

    x = np.arange(len(df))

    z = np.polyfit(
        x,
        df["Close"],
        1
    )

    trendline = np.poly1d(z)(x)

    # =========================================================
    # MACHINE LEARNING PREDICTION
    # =========================================================

    ml_df = df.copy()

    ml_df["Target"] = (
        ml_df["Close"]
        .shift(-1)
    )

    ml_df = ml_df.dropna()

    X = np.array(
        range(len(ml_df))
    ).reshape(-1, 1)

    y = ml_df["Target"].values

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict(
        np.array([[len(ml_df) + 1]])
    )[0]

    # =========================================================
    # LAST DATA
    # =========================================================

    last_price = float(
        df["Close"].iloc[-1]
    )

    rsi = float(
        df["RSI"].iloc[-1]
    )

    # =========================================================
    # SIGNAL
    # =========================================================

    if prediction > last_price and rsi < 70:

        signal = "BUY 🚀"

    elif prediction < last_price and rsi > 30:

        signal = "SELL 🔻"

    else:

        signal = "HOLD ⏳"

    # =========================================================
    # BUY SELL SIGNAL
    # =========================================================

    df["Buy_Signal"] = np.where(
        df["MA20"] > df["MA50"],
        df["Low"] * 0.995,
        np.nan
    )

    df["Sell_Signal"] = np.where(
        df["MA20"] < df["MA50"],
        df["High"] * 1.005,
        np.nan
    )

    # =========================================================
    # BOS / CHOCH
    # =========================================================

    swing_high = (
        df["High"]
        .rolling(5)
        .max()
    )

    swing_low = (
        df["Low"]
        .rolling(5)
        .min()
    )

    df["BOS_UP"] = np.where(
        df["Close"] > swing_high.shift(1),
        df["High"] * 1.01,
        np.nan
    )

    df["BOS_DOWN"] = np.where(
        df["Close"] < swing_low.shift(1),
        df["Low"] * 0.99,
        np.nan
    )

    # =========================================================
    # FIBONACCI
    # =========================================================

    high_price = df["High"].max()

    low_price = df["Low"].min()

    diff = high_price - low_price

    fib_236 = high_price - diff * 0.236
    fib_382 = high_price - diff * 0.382
    fib_500 = high_price - diff * 0.500
    fib_618 = high_price - diff * 0.618

    # =========================================================
    # PROFIT ANALYSIS
    # =========================================================

    take_profit = last_price * 1.03

    cut_loss = last_price * 0.97

    estimated_profit = (
        (
            prediction - last_price
        ) / last_price
    ) * 100

    risk = last_price - cut_loss

    reward = take_profit - last_price

    if risk != 0:

        risk_reward = reward / risk

    else:

        risk_reward = 0

    # =========================================================
    # SMART MONEY
    # =========================================================

    avg_volume = (
        df["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    current_volume = (
        df["Volume"]
        .iloc[-1]
    )

    smart_money = "NEUTRAL"

    if (
        current_volume > avg_volume * 1.5
        and df["Close"].iloc[-1]
        > df["Open"].iloc[-1]
    ):

        smart_money = "ACCUMULATION 🟢"

    elif (
        current_volume > avg_volume * 1.5
        and df["Close"].iloc[-1]
        < df["Open"].iloc[-1]
    ):

        smart_money = "DISTRIBUTION 🔴"

    # =========================================================
    # AI PROBABILITY
    # =========================================================

    bullish_score = 0
    bearish_score = 0

    if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
        bullish_score += 1
    else:
        bearish_score += 1

    if rsi < 30:
        bullish_score += 1

    elif rsi > 70:
        bearish_score += 1

    if prediction > last_price:
        bullish_score += 1
    else:
        bearish_score += 1

    if current_volume > avg_volume:
        bullish_score += 1

    total_score = bullish_score + bearish_score

    if total_score != 0:

        bullish_probability = (
            bullish_score / total_score
        ) * 100

        bearish_probability = (
            bearish_score / total_score
        ) * 100

    else:

        bullish_probability = 0
        bearish_probability = 0

    # =========================================================
    # AI CONFIDENCE
    # =========================================================

    confidence = abs(
        bullish_probability
        - bearish_probability
    )

    if confidence > 70:

        confidence_label = "HIGH CONFIDENCE"

    elif confidence > 55:

        confidence_label = "MEDIUM CONFIDENCE"

    else:

        confidence_label = "LOW CONFIDENCE"

    # =========================================================
    # TELEGRAM BOT
    # =========================================================

    if auto_telegram:

        telegram_message = f"""
📈 AI SAHAM INDONESIA

Saham:
{stock.upper()}

Harga:
Rp {last_price:,.0f}

Prediksi:
Rp {prediction:,.0f}

Signal:
{signal}

Bullish Probability:
{bullish_probability:.2f}%

RSI:
{rsi:.2f}
"""

        success = send_telegram(
            telegram_message
        )

        if success:

            st.success(
                "Telegram signal terkirim"
            )

        else:

            st.error(
                "Telegram gagal terkirim"
            )

    # =========================================================
    # LIVE STATUS
    # =========================================================

    st.success(
        f"🟢 Live Realtime Active ({timeframe})"
    )

    # =========================================================
    # METRICS
    # =========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Harga",
        f"Rp {last_price:,.0f}"
    )

    col2.metric(
        "Prediksi",
        f"Rp {prediction:,.0f}"
    )

    col3.metric(
        "RSI",
        f"{rsi:.2f}"
    )

    col4.metric(
        "Signal",
        signal
    )

    # =========================================================
    # AI DASHBOARD
    # =========================================================

    st.markdown(
        f"""
### 🤖 AI Market Intelligence

- Smart Money: **{smart_money}**
- Bullish Probability: **{bullish_probability:.2f}%**
- Bearish Probability: **{bearish_probability:.2f}%**
- Confidence: **{confidence_label}**
- Estimasi Profit: **{estimated_profit:.2f}%**
- Take Profit: **Rp {take_profit:,.0f}**
- Cut Loss: **Rp {cut_loss:,.0f}**
- Risk Reward: **1 : {risk_reward:.2f}**
"""
    )

    # =========================================================
    # MAIN CHART
    # =========================================================

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            mode="lines",
            name="MA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA50"],
            mode="lines",
            name="MA50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Support"],
            mode="lines",
            name="Support"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Resistance"],
            mode="lines",
            name="Resistance"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=trendline,
            mode="lines",
            name="AI Trendline"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Buy_Signal"],
            mode="markers",
            name="BUY",
            marker=dict(
                symbol="triangle-up",
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Sell_Signal"],
            mode="markers",
            name="SELL",
            marker=dict(
                symbol="triangle-down",
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BOS_UP"],
            mode="markers",
            name="BOS UP",
            marker=dict(
                symbol="star",
                size=10
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BOS_DOWN"],
            mode="markers",
            name="BOS DOWN",
            marker=dict(
                symbol="x",
                size=10
            )
        )
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.2
        )
    )

    # FIBONACCI

    fig.add_hline(y=fib_236)
    fig.add_hline(y=fib_382)
    fig.add_hline(y=fib_500)
    fig.add_hline(y=fib_618)

    # =========================================================
    # CHART LAYOUT
    # =========================================================

    fig.update_layout(
        template="plotly_dark",
        height=900,

        title=f"{stock.upper()} AI Smart Money Chart",

        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        ),

        yaxis=dict(
            title="Price"
        ),

        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),

        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================================================
    # COMPARE CHART
    # =========================================================

    st.subheader("📊 Compare Saham")

    compare_fig = go.Figure()

    compare_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=stock.upper()
        )
    )

    for comp in compare_list:

        try:

            comp_df = load_data(
                comp,
                timeframe
            )

            if comp_df.empty:
                continue

            comp_df["Close"] = pd.to_numeric(
                comp_df["Close"],
                errors="coerce"
            )

            comp_df = comp_df.dropna()

            compare_fig.add_trace(
                go.Scatter(
                    x=comp_df.index,
                    y=comp_df["Close"],
                    mode="lines",
                    name=comp.replace(".JK", "")
                )
            )

        except:
            pass

    compare_fig.update_layout(
        template="plotly_dark",
        height=600,
        title="Perbandingan Saham"
    )

    st.plotly_chart(
        compare_fig,
        use_container_width=True
    )

    # =========================================================
    # REALTIME TABLE
    # =========================================================

    st.subheader("Realtime Data")

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

    # =========================================================
    # DISCLAIMER
    # =========================================================

    st.warning(
        "AI ini hanya untuk edukasi dan bukan financial advice."
    )

# =========================================================
# ERROR HANDLER
# =========================================================

except Exception as e:

    st.error(
        f"Terjadi error: {str(e)}"
    )