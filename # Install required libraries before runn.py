import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# --- Custom Page Config ---
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3 {
            color: #f5f5f5;
        }
        .stButton button {
            background: linear-gradient(135deg, #1f77b4, #00c853);
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #00c853, #1f77b4);
            color: black;
        }
        .reportview-container .markdown-text-container {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("📈 Stock Price Prediction Dashboard")
st.write("Predict future stock prices using **Facebook Prophet** with interactive charts.")

# --- Sidebar for Inputs ---
st.sidebar.header("🔧 Settings")
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
n_days = st.sidebar.slider("Forecast Days", 30, 365, 90)
st.sidebar.markdown("💡 Examples: `AAPL`, `TSLA`, `GOOGL`, `RELIANCE.NS`")

if st.sidebar.button("🚀 Run Prediction"):
    st.subheader(f"Fetching data for **{stock_symbol}**...")
    data = yf.download(stock_symbol, start="2020-01-01")
    if data.empty:
        st.error(f"No data found for {stock_symbol}. Please check the ticker symbol.")
        st.stop()

    # Prepare Data
    df = data.reset_index()
    if 'Date' in df.columns:
        df = df[['Date', 'Close']]
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    elif 'Datetime' in df.columns:
        df = df[['Datetime', 'Close']]
        df.rename(columns={'Datetime': 'ds', 'Close': 'y'}, inplace=True)
    else:
        st.error("Downloaded data does not contain 'Date' or 'Datetime' columns.")
        st.stop()

    try:
        # Build Prophet Model
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        # Forecast
        future = model.make_future_dataframe(periods=n_days)
        forecast = model.predict(future)

        # --- Layout: 2 Columns ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Recent Stock Data")
            st.dataframe(df.tail(10))
        with col2:
            st.subheader("📅 Forecasted Prices (Last Few Days)")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

        # --- Interactive Chart ---
        st.subheader("📈 Stock Price Forecast")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Confidence Interval Chart ---
        st.subheader("🔮 Forecast with Confidence Interval")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                  name="Predicted", line=dict(color='cyan')))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                  name="Upper Bound", line=dict(color='green', dash='dot')))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                  name="Lower Bound", line=dict(color='red', dash='dot')))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

