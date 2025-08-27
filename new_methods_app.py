import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Streamlit Page Config
st.set_page_config(page_title="Temperature Prediction", layout="wide")

# Title
st.title("🌡️ Temperature Prediction App")

# File Upload
uploaded_file = st.file_uploader("Upload your temperature dataset (CSV)", type=["csv"])

if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    if 'YEAR' not in df.columns or 'ANNUAL' not in df.columns:
        st.error("Dataset must contain 'YEAR' and 'ANNUAL' columns.")
        st.stop()

    # Convert to numeric and handle invalid entries
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['ANNUAL'] = pd.to_numeric(df['ANNUAL'], errors='coerce')

    # Handle missing values globally
    if df.isnull().sum().sum() > 0:
        st.warning(f"Found {df.isnull().sum().sum()} missing values. Filling with column mean.")
        df.fillna(df.mean(numeric_only=True), inplace=True)

    # Validate if dataset has enough data
    if len(df) < 10:
        st.error("Dataset must have at least 10 rows for meaningful prediction.")
        st.stop()

    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Sidebar Inputs
    st.sidebar.header("Prediction Options")
    method = st.sidebar.selectbox(
        "Select Prediction Method",
        ["1. ARIMA", "2. Linear Regression", "3. Random Forest", "4. LSTM"]
    )

    predict_year = st.sidebar.number_input(
        "Enter the year to predict for:",
        min_value=int(df['YEAR'].max()) + 1,
        value=int(df['YEAR'].max()) + 1
    )

    # Prepare Data
    X = df[['YEAR']]
    y = df['ANNUAL']

    # Prediction
    if st.sidebar.button("Predict"):
        st.subheader(f"Prediction for year {predict_year} using {method}")
        prediction = None

        # ----- METHOD 1: ARIMA -----
        if "ARIMA" in method:
            try:
                y_series = pd.Series(y.values, index=df['YEAR'])
                steps = predict_year - int(df['YEAR'].max())
                if steps <= 0:
                    st.error("Prediction year must be greater than the last year in the dataset.")
                else:
                    model = ARIMA(y_series, order=(2, 1, 2))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=steps)
                    prediction = forecast.iloc[-1]
            except Exception as e:
                st.error(f"ARIMA model error: {e}")

        # ----- METHOD 2: Linear Regression -----
        elif "Linear Regression" in method:
            try:
                lr = LinearRegression()
                lr.fit(X, y)
                prediction = lr.predict([[predict_year]])[0]
            except Exception as e:
                st.error(f"Linear Regression error: {e}")

        # ----- METHOD 3: Random Forest -----
        elif "Random Forest" in method:
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                prediction = rf.predict([[predict_year]])[0]
            except Exception as e:
                st.error(f"Random Forest error: {e}")

        # ----- METHOD 4: LSTM -----
        elif "LSTM" in method:
            try:
                seq_length = 5
                data = y.values.reshape(-1, 1)
                generator = TimeseriesGenerator(data, data, length=seq_length, batch_size=1)

                # Build LSTM Model
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')

                # Train model
                model.fit(generator, epochs=50, verbose=0)

                # Predict next values
                steps = predict_year - int(df['YEAR'].max())
                if steps <= 0:
                    st.error("Prediction year must be greater than the last year in the dataset.")
                else:
                    last_seq = data[-seq_length:]
                    for _ in range(steps):
                        last_seq_reshaped = last_seq.reshape((1, seq_length, 1))
                        next_value = model.predict(last_seq_reshaped, verbose=0)[0][0]
                        last_seq = np.append(last_seq[1:], next_value)
                    prediction = last_seq[-1]
            except Exception as e:
                st.error(f"LSTM model error: {e}")

        # Show Prediction
        if prediction is not None:
            st.success(f"Predicted Annual Temperature for {predict_year}: **{round(prediction, 2)}°C**")
