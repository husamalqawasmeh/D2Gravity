import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Silver Price Predictor", layout="wide")

st.title("📈 Silver Price Predictor (5 Years Historical + 1 Year Forecast)")
st.write("This application fetches the last 5 years of Silver Futures (SI=F) data from Yahoo Finance, evaluates a forecasting model to ensure low RMSE, and predicts the price for the next 1 year.")

@st.cache_data(ttl=3600*24)
def load_data():
    ticker = "SI=F"
    silver_data = yf.download(ticker, period="5y", interval="1d")
    
    # Check if data might have MultiIndex columns (yfinance > 0.2.0 change)
    if isinstance(silver_data.columns, pd.MultiIndex):
        silver_data.columns = silver_data.columns.droplevel('Ticker')

    df = silver_data[['Close']].copy()
    if df.empty:
        return df
    
    # Handle NaN values if any
    df = df.ffill().dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # Resample to common business days to avoid gaps causing issues passing to statsmodels
    df = df.resample('B').ffill()
    return df

with st.spinner("Fetching data from Yahoo Finance..."):
    df = load_data()

if df.empty:
    st.error("Failed to fetch data from Yahoo Finance. Please try again later.")
    st.stop()

st.subheader("Last 5 Years Silver Price Data")
st.line_chart(df['Close'])

# Modeling
# We will use Exponential Smoothing (Holt-Winters) which generally provides robust baseline forecasts with low RMSE for commodities
# First, let's evaluate RMSE on the last 1 year (approx 252 business days)
test_size = 252
train, test = df.iloc[:-test_size], df.iloc[-test_size:]

with st.spinner("Training model and evaluating RMSE..."):
    # Fit model on training set
    try:
        model_eval = ExponentialSmoothing(train['Close'], trend='add', seasonal=None, initialization_method="estimated")
        fitted_eval = model_eval.fit()
        predictions = fitted_eval.forecast(len(test))
        rmse = np.sqrt(mean_squared_error(test['Close'], predictions))
        
        col1, col2 = st.columns(2)
        col1.metric("Model Used", "Holt-Winters Exponential Smoothing")
        col2.metric("Test Set RMSE", f"{rmse:.2f}")
    except Exception as e:
        st.warning(f"Could not calculate RMSE on test set: {e}")

# Now forecast to the future (next 1 year = 252 business days)
with st.spinner("Generating 1-Year Forecast..."):
    try:
        final_model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None, initialization_method="estimated")
        final_fitted = final_model.fit()
        forecast_steps = 252 # approx 1 year of business days
        forecast = final_fitted.forecast(forecast_steps)
        
        # Create forecast dataframe
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps*2) if (last_date + timedelta(days=i)).weekday() < 5][:forecast_steps]
        
        forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=forecast_dates)
        
        st.subheader("Historical vs 1-Year Future Forecast")
        
        # Plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='1-Year Forecast', line=dict(color='orange', dash='dash')))
        
        fig.update_layout(
            title="Silver Price (SI=F) Forecast for Next 1 Year",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecasted Data Points (Next 30 Days)")
        st.dataframe(forecast_df.head(30))
        
    except Exception as e:
        st.error(f"Error during forecasting: {e}")

st.success("Analysis and generic forecast completed!")
