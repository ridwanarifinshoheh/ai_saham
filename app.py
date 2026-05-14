# =========================================================
# AI SAHAM INDONESIA ULTRA PRO MAX
# REALTIME + MOBILE + TELEGRAM BOT
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# =========================================================
# AUTO REFRESH REALTIME
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
# MOBILE RESPONSIVE CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp{
        background:#0d1117;
        color:white;
    }

    div[data-testid="metric-container"]{
        background:#161b22;
        border:1px solid #30363d;
        padding:15px;
        border-radius:15px;
    }

    /* MOBILE */
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
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["15m", "1h", "1d"],
    index=1
)

auto_telegram = st.sidebar.toggle(
    "Telegram Signal Bot",
    value=False
)

# =========================================================
# INPUT
# =========================================================

stock = st.text_input(
    "Kode Saham",
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

    data = yf.download(
        code,
        period="3mo",
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

# =========================================================
# TELEGRAM FUNCTION
# =========================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    requests.post(url, data=payload)

# =========================================================
# MAIN
# =========================================================

try:

    # =========================================================
    # GET DATA
    # =========================================================

    df = load_data(stock_code, timeframe)

    if df.empty:
        st.error("Data saham tidak ditemukan")
        st.stop()

    # =========================================================
    # CLEAN DATA
    # =========================================================

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    # =========================================================
    # MOVING AVERAGE
    # =========================================================

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    # =========================================================
    # RSI
    # =========================================================

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # =========================================================
    # SUPPORT RESISTANCE
    # =========================================================

    df["Support"] = df["Low"].rolling(20).min()

    df["Resistance"] = df["High"].rolling(20).max()

    # =========================================================
    # TRENDLINE
    # =========================================================

    x = np.arange(len(df))

    z = np.polyfit(x, df["Close"], 1)

    trendline = np.poly1d(z)(x)

    # =========================================================
    # AI PREDICTION
    # =========================================================

    ml_df = df.copy()

    ml_df["Target"] = ml_df["Close"].shift(-1)

    ml_df = ml_df.dropna()

    X = np.array(range(len(ml_df))).reshape(-1, 1)

    y = ml_df["Target"].values

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict(
        np.array([[len(ml_df) + 1]])
    )[0]

    # =========================================================
    # LAST DATA
    # =========================================================

    last_price = float(df["Close"].iloc[-1])

    rsi = float(df["RSI"].iloc[-1])

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
    # TELEGRAM SIGNAL
    # =========================================================

    if auto_telegram:

        telegram_message = f"""

📈 AI SAHAM INDONESIA

Saham: {stock.upper()}

Harga:
Rp {last_price:,.0f}

Prediksi:
Rp {prediction:,.0f}

Signal:
{signal}

RSI:
{rsi:.2f}

        """

        send_telegram(telegram_message)

        st.success("Telegram signal terkirim")

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
    # REALTIME STATUS
    # =========================================================

    st.success(
        "🟢 Realtime AI Dashboard Active"
    )

    # =========================================================
    # MAIN CHART
    # =========================================================

    fig = go.Figure()

    # CANDLE
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

    # MA20
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            mode="lines",
            name="MA20"
        )
    )

    # MA50
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA50"],
            mode="lines",
            name="MA50"
        )
    )

    # SUPPORT
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Support"],
            mode="lines",
            name="Support"
        )
    )

    # RESISTANCE
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Resistance"],
            mode="lines",
            name="Resistance"
        )
    )

    # TRENDLINE
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=trendline,
            mode="lines",
            name="AI Trendline"
        )
    )

    # VOLUME
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.2
        )
    )

    # =========================================================
    # CHART LAYOUT
    # =========================================================

    fig.update_layout(
        template="plotly_dark",
        height=850,

        title=f"{stock.upper()} AI Live Chart",

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

            comp_df = load_data(comp, "1d")

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
    # DATA TABLE
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