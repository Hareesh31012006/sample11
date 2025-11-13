import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import os
import joblib
import matplotlib.pyplot as plt
import time

# ML / DL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nltk

# ---------------------------
# Auto-download TextBlob corpora
# ---------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    with st.spinner("Downloading TextBlob corpora..."):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('brown')
        nltk.download('movie_reviews')
        nltk.download('vader_lexicon')
    st.success("TextBlob corpora downloaded successfully!")

# ---------------------------
# Utility & Caching functions
# ---------------------------

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily historical data from Yahoo Finance with retry and fallback to CSV."""
    csv_path = f"{ticker.upper()}_fallback.csv"

    # Try local CSV first
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            df = df.loc[start:end] if start and end else df
            df = df.resample('D').mean().interpolate('linear')
            st.warning(f"Using local CSV fallback for {ticker}.")
            return df
        except Exception as e:
            st.error(f"Failed to load local CSV for {ticker}: {e}")

    # Try Yahoo Finance up to 3 times
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                df = df[['Close']].resample('D').mean().interpolate('linear')
                df.to_csv(csv_path)  # save fallback CSV
                return df
        except Exception as e:
            st.write(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    st.error(f"No market data available for {ticker}.")
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_news_sentiment(ticker: str, dates_index: pd.DatetimeIndex) -> pd.Series:
    """Fetch news sentiment via yfinance; fallback to neutral if unavailable."""
    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news
        if raw_news:
            rows = []
            for item in raw_news:
                title = item.get("title", "")
                pub_time = item.get("providerPublishTime")
                if pub_time is None or title is None:
                    continue
                pub_dt = datetime.fromtimestamp(pub_time)
                polarity = TextBlob(title).sentiment.polarity
                rows.append({"date": pub_dt.date(), "polarity": polarity})
            if rows:
                news_df = pd.DataFrame(rows)
                daily_sentiment = news_df.groupby('date').polarity.mean()
                sentiment_series = pd.Series(0.0, index=dates_index)
                for idx in dates_index:
                    sentiment_series.loc[idx] = daily_sentiment.get(idx.date(), 0.0)
                return sentiment_series
    except Exception:
        st.write("Note: could not fetch news via yfinance; using neutral sentiment fallback.")
    return pd.Series(0.0, index=dates_index)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra features if needed (currently just Close + Sentiment)."""
    return df

def create_sequences_multivariate(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, :])
        y.append(arr[i, 0])  # predict Close only
    return np.array(X), np.array(y)

@st.cache_resource
def build_model(input_shape, lr=5e-4, lstm_units=64):
    """Build Bi-LSTM model."""
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
st.title("📈 BI-LSTM Stock Analyzer with Sentiment")

# Sidebar
with st.sidebar:
    st.header("Analyze a stock")
    ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
    start_date = st.date_input("Start date", value=(datetime.today() - timedelta(days=365*2)).date())
    end_date = st.date_input("End date", value=datetime.today().date())
    lookback = st.slider("Lookback (days)", 7, 180, 30)
    epochs = st.slider("Epochs", 10, 300, 50, 10)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    retrain = st.checkbox("Force retrain model", value=False)
    analyze_button = st.button("🔎 Analyse")

col1, col2 = st.columns([2, 1])

if analyze_button:
    if not ticker:
        st.warning("Enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_stock_data(ticker.upper(), start_date.isoformat(), end_date.isoformat())
        if df.empty:
            st.error(f"No market data for {ticker}.")
        else:
            sentiment = fetch_news_sentiment(ticker.upper(), df.index)
            df2 = df.copy()
            df2['Sentiment'] = sentiment.values
            df2 = create_features(df2)

            with col1:
                st.subheader(f"{ticker.upper()} - Recent Data")
                st.write(df2.tail(10))

            scaler_close = MinMaxScaler()
            scaler_sentiment = MinMaxScaler()
            df_scaled = df2.copy()
            df_scaled['Close'] = scaler_close.fit_transform(df2[['Close']])
            df_scaled['Sentiment'] = scaler_sentiment.fit_transform(df2[['Sentiment']])

            X, y = create_sequences_multivariate(df_scaled.values, lookback)
            if len(X) < 10:
                st.error("Not enough data after lookback.")
            else:
                train_size = int(len(X)*0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                input_shape = (X_train.shape[1], X_train.shape[2])
                model_path = f"model_{ticker.upper()}_lb{lookback}.h5"
                scaler_close_path = f"scaler_close_{ticker.upper()}_lb{lookback}.joblib"
                scaler_sent_path = f"scaler_sent_{ticker.upper()}_lb{lookback}.joblib"

                if retrain:
                    build_model.clear()
                    for f in [model_path, scaler_close_path, scaler_sent_path]:
                        if os.path.exists(f):
                            os.remove(f)

                model = build_model(input_shape)

                loaded_from_file = False
                if os.path.exists(model_path) and not retrain:
                    try:
                        model.load_weights(model_path)
                        loaded_from_file = True
                    except Exception:
                        loaded_from_file = False

                if not loaded_from_file:
                    with st.spinner("Training Bi-LSTM model..."):
                        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        model.fit(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[es],
                            verbose=0
                        )
                        model.save_weights(model_path)
                        joblib.dump(scaler_close, scaler_close_path)
                        joblib.dump(scaler_sentiment, scaler_sent_path)

                preds_scaled = model.predict(X_test)
                preds_inv = scaler_close.inverse_transform(preds_scaled)
                actual_inv = scaler_close.inverse_transform(y_test.reshape(-1,1))

                mae = mean_absolute_error(actual_inv, preds_inv)
                rmse = np.sqrt(mean_squared_error(actual_inv, preds_inv))

                with col1:
                    st.subheader("Model Performance (Test Set)")
                    st.write(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(actual_inv, label='Actual')
                    ax.plot(preds_inv, label='Predicted')
                    ax.legend()
                    st.pyplot(fig)

                last_seq = df_scaled.values[-lookback:].reshape(1, lookback, df_scaled.shape[1])
                next_scaled = model.predict(last_seq)
                next_close = scaler_close.inverse_transform(next_scaled)[0,0]
                last_actual = df2['Close'].iloc[-1]

                rec_label, pct_change = get_recommendation(last_actual, next_close, threshold_pct=0.01)

                with col2:
                    st.metric("Last Close", f"{last_actual:.2f}")
                    st.metric("Predicted Next Close", f"{next_close:.2f}", delta=f"{pct_change*100:.2f}%")
                    st.markdown(f"### Recommendation: **{rec_label}**")
                    if rec_label == "BUY":
                        st.success("Model suggests BUY.")
                    elif rec_label == "SELL":
                        st.error("Model suggests SELL.")
                    else:
                        st.info("Model suggests HOLD.")

                    st.subheader("Recent Sentiment")
                    st.line_chart(df2['Sentiment'].tail(60))

                with st.expander("Download model & scalers"):
                    if os.path.exists(model_path):
                        st.download_button("Download weights (.h5)", data=open(model_path, "rb").read(), file_name=model_path)
                    if os.path.exists(scaler_close_path):
                        st.download_button("Download Close scaler (.joblib)", data=open(scaler_close_path, "rb").read(), file_name=scaler_close_path)
                    if os.path.exists(scaler_sent_path):
                        st.download_button("Download Sentiment scaler (.joblib)", data=open(scaler_sent_path, "rb").read(), file_name=scaler_sent_path)

                st.success("Analysis complete.")
