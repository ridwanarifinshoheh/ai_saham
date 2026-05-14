import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ======================================================
# AUTO REFRESH
# ======================================================

st_autorefresh(interval=10000, key="refresh")

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Saham Indonesia PRO",
    layout="wide"
)

# ======================================================
# PREMIUM DARK MODE
# ======================================================

st.markdown(
    """
    <style>

    .stApp{
        background-color:#0d1117;
        color:white;
    }

    div[data-testid="metric-container"]{
        background-color:#161b22;
        border:1px solid #30363d;
        padding:15px;
        border-radius:15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# TITLE
# ======================================================

st.title("📈 AI Saham Indonesia PRO")
st.caption("Live Smart Money Trading Dashboard")

# ======================================================
# INPUT
# ======================================================

stock = st.text_input(
    "Kode Saham Utama",
    "BBCA"
)

compare_stocks = st.text_input(
    "Compare Saham (pisahkan koma)",
    "BBRI,TLKM"
)

stock_code = stock.upper() + ".JK"

compare_list = [
    s.strip().upper() + ".JK"
    for s in compare_stocks.split(",")
]

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data(ttl=10)
def load_data(code):

    data = yf.download(
        code,
        period="3mo",
        interval="1h",
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


@st.cache_data(ttl=10)
def load_compare_data(code):

    data = yf.download(
        code,
        period="3mo",
        interval="1d",
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

# ======================================================
# MAIN
# ======================================================

try:

    df = load_data(stock_code)

    if df.empty:
        st.error("Data saham tidak ditemukan")
        st.stop()

    # ======================================================
    # CLEAN DATA
    # ======================================================

    for col in ["Open", "High", "Low", "Close", "Volume"]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    # ======================================================
    # MOVING AVERAGE
    # ======================================================

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    # ======================================================
    # RSI
    # ======================================================

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # ======================================================
    # SUPPORT & RESISTANCE
    # ======================================================

    df["Support"] = df["Low"].rolling(20).min()

    df["Resistance"] = df["High"].rolling(20).max()

    # ======================================================
    # AI TRENDLINE
    # ======================================================

    x = np.arange(len(df))

    z = np.polyfit(x, df["Close"], 1)

    trendline = np.poly1d(z)(x)

    # ======================================================
    # MACHINE LEARNING
    # ======================================================

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

    # ======================================================
    # LAST DATA
    # ======================================================

    last_price = float(df["Close"].iloc[-1])

    rsi = float(df["RSI"].iloc[-1])

    # ======================================================
    # SIGNAL
    # ======================================================

    if prediction > last_price and rsi < 70:

        signal = "BUY 🚀"
        signal_color = "lime"

    elif prediction < last_price and rsi > 30:

        signal = "SELL 🔻"
        signal_color = "red"

    else:

        signal = "HOLD ⏳"
        signal_color = "orange"

    # ======================================================
    # BUY SELL SIGNAL
    # ======================================================

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

    # ======================================================
    # METRICS
    # ======================================================

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

    st.markdown(
        f"""
        <h2 style='color:{signal_color};'>
        {signal}
        </h2>
        """,
        unsafe_allow_html=True
    )

    # ======================================================
    # MAIN CHART
    # ======================================================

    fig = go.Figure()

    # CANDLESTICK
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
            name="Support",
            line=dict(dash="dot")
        )
    )

    # RESISTANCE
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Resistance"],
            mode="lines",
            name="Resistance",
            line=dict(dash="dot")
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

    # BUY SIGNAL
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Buy_Signal"],
            mode="markers",
            name="BUY",
            marker=dict(
                symbol="triangle-up",
                size=12
            )
        )
    )

    # SELL SIGNAL
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Sell_Signal"],
            mode="markers",
            name="SELL",
            marker=dict(
                symbol="triangle-down",
                size=12
            )
        )
    )

    # VOLUME
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.25
        )
    )

    # ======================================================
    # LAYOUT
    # ======================================================

    fig.update_layout(
        template="plotly_dark",
        height=850,
        title=f"{stock.upper()} Live AI Chart",

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

    # ======================================================
    # COMPARE CHART
    # ======================================================

    st.subheader("📊 Compare Saham")

    compare_fig = go.Figure()

    # MAIN STOCK
    compare_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=stock.upper()
        )
    )

    # COMPARE STOCKS
    for comp in compare_list:

        try:

            comp_df = load_compare_data(comp)

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
        title="Perbandingan Saham",
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(
        compare_fig,
        use_container_width=True
    )

    # ======================================================
    # DATA TABLE
    # ======================================================

    st.subheader("Realtime Data")

    st.dataframe(
        df.tail(20)
    )

    # ======================================================
    # DISCLAIMER
    # ======================================================

    st.warning(
        "AI ini hanya untuk edukasi dan bukan financial advice."
    )

except Exception as e:

    st.error(
        f"Terjadi error: {e}"
    )