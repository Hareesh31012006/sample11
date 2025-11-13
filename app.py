"""
Streamlit Stock Analyzer with BI-LSTM + Sentiment
- Enter a ticker, click 'Analyse' to fetch daily data (yfinance) + sentiment (TextBlob)
- Trains a Bi-LSTM model on the last N days, predicts next-day(s)
- Shows plots and gives a simple Buy/Hold/Sell suggestion
"""
import nltk
nltk.download('punkt', quiet=True)


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import os
import joblib

# ML / DL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Utility & Caching functions
# ---------------------------

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetches daily historical data from Yahoo Finance and resamples to business days (D).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df[['Close']].rename(columns={'Close': 'Close'})
    # Ensure daily freq & fill small gaps
    df = df.resample('D').mean().interpolate(method='linear')
    return df

@st.cache_data(show_spinner=False)
def fetch_news_sentiment(ticker: str, dates_index: pd.DatetimeIndex) -> pd.Series:
    """
    Try to get news via yfinance Ticker.news (may be available). If not, we fallback
    to a simple neutral sentiment for all days (or trivial proxy).
    We build daily sentiment scores aligned with dates_index.
    """
    try:
        tk = yf.Ticker(ticker)
        news_items = []
        # .news returns dicts with 'title' and 'publisher' and 'link' — availability varies by ticker
        raw_news = tk.news
        if raw_news:
            # Build a daily sentiment by mapping each news piece to its published date
            for item in raw_news:
                title = item.get("title", "")
                provider_pub_time = item.get("providerPublishTime", None)
                if provider_pub_time:
                    pub_dt = datetime.fromtimestamp(provider_pub_time)
                else:
                    pub_dt = None
                news_items.append((pub_dt, title))
        # If we have news, compute polarity and sum/avg per date
        if news_items:
            # Create df
            rows = []
            for pub_dt, title in news_items:
                if title is None:
                    continue
                if pub_dt is None:
                    # if no date, skip or attempt to use today
                    pub_dt = datetime.utcnow()
                polarity = TextBlob(title).sentiment.polarity
                rows.append({"date": pub_dt.date(), "polarity": polarity})
            news_df = pd.DataFrame(rows)
            if news_df.empty:
                raise ValueError("No usable news rows")
            # Group by date and average polarity
            daily = news_df.groupby('date').polarity.mean()
            # Align with provided index -> map daily polarity to each date; fill missing with 0
            sentiment_series = pd.Series(0.0, index=dates_index)
            for idx in dates_index:
                pol = daily.get(idx.date(), 0.0)
                sentiment_series.loc[idx] = pol
            return sentiment_series
    except Exception as e:
        # If anything fails (no news endpoint, network, API changes), fall back to neutral
        st.write("Note: could not fetch news via yfinance; using neutral sentiment as fallback.")
    # fallback: neutral sentiment series (zeros)
    return pd.Series(0.0, index=dates_index)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features if needed. Right now we keep Close and Sentiment.
    """
    return df

def create_sequences_multivariate(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, :])
        y.append(arr[i, 0])  # predict Close (first column)
    return np.array(X), np.array(y)

@st.cache_resource
def build_model(input_shape, lr=5e-4, lstm_units=64):
    """
    Create and return the Bi-LSTM model. Cached so rebuilding is cheap.
    """
    model = Sequential([
        Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(lstm_units)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def get_recommendation(last_price, predicted_next_price, threshold_pct=0.01):
    """
    Simple rule:
      - If predicted increase > threshold_pct (e.g., 1%) -> BUY
      - If predicted decrease < -threshold_pct -> SELL
      - Else HOLD
    """
    pct_change = (predicted_next_price - last_price) / last_price
    if pct_change > threshold_pct:
        return "BUY", pct_change
    elif pct_change < -threshold_pct:
        return "SELL", pct_change
    else:
        return "HOLD", pct_change

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="BI-LSTM Stock Analyzer", layout="wide")
st.title("📈 BI-LSTM Daily Stock Analyzer (with Sentiment)")

# Sidebar controls
with st.sidebar:
    st.header("Analyze a stock")
    ticker = st.text_input("Enter ticker (e.g., AAPL)", value="AAPL")
    start_date = st.date_input("Start date", value=(datetime.today() - timedelta(days=365*2)).date())
    end_date = st.date_input("End date", value=(datetime.today()).date())
    lookback = st.slider("Lookback (days) for model", min_value=7, max_value=180, value=30, step=1)
    epochs = st.slider("Epochs (training)", min_value=10, max_value=300, value=50, step=10)
    batch_size = st.selectbox("Batch size", options=[16, 32, 64], index=1)
    retrain = st.checkbox("Force retrain model (ignore cached)", value=False)
    analyze_button = st.button("🔎 Analyse")

col1, col2 = st.columns([2, 1])

if analyze_button:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching {ticker} data from Yahoo Finance..."):
            df = fetch_stock_data(ticker.upper(), start=start_date.isoformat(), end=end_date.isoformat())
        if df.empty:
            st.error(f"No market data found for {ticker}. Check ticker symbol or date range.")
        else:
            # Add sentiment: try news-based; fallback neutral
            sentiment = fetch_news_sentiment(ticker.upper(), df.index)
            df2 = df.copy()
            df2['Sentiment'] = sentiment.values
            df2 = create_features(df2)

            # Display data snippet
            with col1:
                st.subheader(f"{ticker.upper()} - Data snapshot")
                st.write(df2.tail(10))

            # Scaling
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df2.values)  # close and sentiment columns

            # Create sequences
            X, y = create_sequences_multivariate(scaled, lookback)
            if len(X) < 10:
                st.error("Not enough data after lookback to train the model. Try a shorter lookback or longer date range.")
            else:
                # Train / test split
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                input_shape = (X_train.shape[1], X_train.shape[2])
                model_cache_path = f"model_{ticker.upper()}_lb{lookback}.h5"
                scaler_path = f"scaler_{ticker.upper()}_lb{lookback}.joblib"

                # Optionally clear cached model resource if forced retrain
                if retrain:
                    try:
                        # remove cached model file if exists
                        if os.path.exists(model_cache_path):
                            os.remove(model_cache_path)
                        if os.path.exists(scaler_path):
                            os.remove(scaler_path)
                    except Exception:
                        pass
                    # Also clear build_model cache by reloading module-level function? We use st.cache_resource so it persists per session.

                # Build model (cached resource)
                model = build_model(input_shape=input_shape, lr=5e-4, lstm_units=64)

                # If a saved model exists and not forcing retrain, load it.
                if os.path.exists(model_cache_path) and not retrain:
                    try:
                        model.load_weights(model_cache_path)
                        loaded_from_file = True
                    except Exception:
                        loaded_from_file = False
                else:
                    loaded_from_file = False

                # Train if required
                if not loaded_from_file:
                    with st.spinner("Training Bi-LSTM model (this may take a few minutes)..."):
                        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[es],
                            verbose=0
                        )
                        # Save weights and scaler for future fast reload
                        model.save_weights(model_cache_path)
                        joblib.dump(scaler, scaler_path)

                # Predictions on test set
                preds_scaled = model.predict(X_test)
                # Build inverse transform shape: we predicted Close (col 0) only; scaler expects full feature shape.
                placeholder = np.zeros((len(preds_scaled), df2.shape[1]))
                placeholder[:, 0] = preds_scaled.flatten()  # set predicted close in col0
                preds_inv = scaler.inverse_transform(placeholder)[:, 0]

                actual_scaled_placeholder = np.zeros((len(y_test), df2.shape[1]))
                actual_scaled_placeholder[:, 0] = y_test
                actual_inv = scaler.inverse_transform(actual_scaled_placeholder)[:, 0]

                # Metrics
                mae = mean_absolute_error(actual_inv, preds_inv)
                mse = mean_squared_error(actual_inv, preds_inv)
                rmse = np.sqrt(mse)

                # Show metrics and plots
                with col1:
                    st.subheader("Model performance (on test set)")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

                    st.subheader("Actual vs Predicted (test set)")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(actual_inv, label='Actual')
                    ax.plot(preds_inv, label='Predicted')
                    ax.set_title(f"{ticker.upper()} Actual vs Predicted (last {len(actual_inv)} points)")
                    ax.legend()
                    st.pyplot(fig)

                # Single-step forecasting: predict next 1 day using last available sequence
                last_seq = scaled[-lookback:, :].reshape(1, lookback, df2.shape[1])
                next_scaled = model.predict(last_seq)  # predicted scaled close
                future_placeholder = np.zeros((1, df2.shape[1]))
                future_placeholder[0, 0] = next_scaled[0, 0]
                next_price = scaler.inverse_transform(future_placeholder)[0, 0]

                last_actual_price = df2['Close'].iloc[-1]

                rec_label, pct_change = get_recommendation(last_actual_price, next_price, threshold_pct=0.01)

                # Display forecast, recommendation
                with col2:
                    st.metric(label="Last Close", value=f"{last_actual_price:.2f}")
                    st.metric(label="Predicted Next Close (1-day)", value=f"{next_price:.2f}",
                              delta=f"{pct_change*100:.2f}%")
                    st.markdown(f"### Recommendation: **{rec_label}**")
                    if rec_label == "BUY":
                        st.success("Model suggests BUY (expected > +1% gain).")
                    elif rec_label == "SELL":
                        st.error("Model suggests SELL (expected > -1% loss).")
                    else:
                        st.info("Model suggests HOLD (small expected change).")

                # Show recent sentiment trace
                with col2:
                    st.subheader("Recent Sentiment (sample)")
                    st.line_chart(df2['Sentiment'].tail(60))

                # Offer to download model weights and scaler if desired
                with st.expander("Download trained model & scaler"):
                    if os.path.exists(model_cache_path):
                        st.write("Model weights saved at:", model_cache_path)
                        st.download_button("Download weights (.h5)", data=open(model_cache_path, "rb").read(),
                                           file_name=model_cache_path)
                    if os.path.exists(scaler_path):
                        st.write("Scaler saved at:", scaler_path)
                        st.download_button("Download scaler (.joblib)", data=open(scaler_path, "rb").read(),
                                           file_name=scaler_path)

                st.success("Analysis complete.")
