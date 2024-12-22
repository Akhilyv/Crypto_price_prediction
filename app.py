import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("E:\\Files\\UB Files\\Programming DB Data Science\\Project\\Final Files\\final_model.joblib")

# Load historical data (used for calculating rolling averages and lag features)
historical_data_path = "E:\\Files\\UB Files\\Programming DB Data Science\\Project\\Final Files\\BTC.csv"  
historical_data = pd.read_csv(historical_data_path)

st.title("Bitcoin Price Movement Prediction")

st.write("""
Provide the following details to predict whether the Bitcoin price will increase or decrease.
The model dynamically calculates rolling averages, lag features, and also predicts the closing price.
""")

# Input fields
open_price = st.number_input("Open Price", min_value=0.0, step=0.01)
high_price = st.number_input("High Price", min_value=0.0, step=0.01)
low_price = st.number_input("Low Price", min_value=0.0, step=0.01)
adj_close_price = st.number_input("Adjusted Close Price", min_value=0.0, step=0.01)
volume = st.number_input("Volume", min_value=0.0, step=0.01)

if st.button("Predict Price Movement and Closing Price"):
    # Append the new data to historical data
    new_data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "AdjClose": adj_close_price,
        "Volume": volume
    }
    historical_data.loc[len(historical_data)] = new_data

    # Calculate additional features
    rolling_mean_volume = historical_data["Volume"].rolling(window=5).mean().iloc[-1]
    rolling_mean_close = historical_data["AdjClose"].rolling(window=5).mean().iloc[-1]
    lag_1_close = historical_data["AdjClose"].shift(1).iloc[-1]
    lag_2_close = historical_data["AdjClose"].shift(2).iloc[-1]

    # Prepare the feature array for the model
    features = np.array([
        [
            open_price,
            high_price,
            low_price,
            adj_close_price,
            volume,
            rolling_mean_volume,
            rolling_mean_close,
            lag_1_close,
            lag_2_close
        ]
    ])

    # Make the prediction
    prediction = model.predict(features)[0]

    # Calculate predicted closing price (simplified for demonstration)
    # This assumes you want to compute the next day's closing price based on historical data
    predicted_close_price = rolling_mean_close  # Simplified; replace with actual regression model if needed

    # Display the prediction result
    st.write("## Results:")
    if prediction == 1:
        st.write("### The model predicts: **Bitcoin price will increase.**")
    else:
        st.write("### The model predicts: **Bitcoin price will decrease.**")

    # Display the predicted closing price
    st.write(f"### Predicted Closing Price: **${predicted_close_price:.2f}**")

    # Display the calculated rolling averages and lag features
    st.write("### Calculated Features:")
    st.write(f"- Rolling Mean Volume (5-day): **{rolling_mean_volume:.2f}**")
    st.write(f"- Rolling Mean Adjusted Close (5-day): **{rolling_mean_close:.2f}**")
    st.write(f"- Lag 1 Adjusted Close: **{lag_1_close:.2f}**")
    st.write(f"- Lag 2 Adjusted Close: **{lag_2_close:.2f}**")
