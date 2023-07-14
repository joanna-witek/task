import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


def predict_sales(model, data):
    prediction = model.predict(data)
    return prediction


def main():
    # Load the trained model
    # model = RandomForestRegressor()
    model = joblib.load('model.h5')

    # Streamlit app title
    st.title('Weekly Sales Prediction')

    # User input for store, department, week, and year
    store = st.number_input('Store', min_value=1, max_value=45, step=1)
    dept = st.number_input('Department', min_value=1, max_value=99, step=1)
    type = st.number_input('Type', min_value=1, max_value=3, step=1)
    week = st.number_input('Week', min_value=1, max_value=52, step=1)
    year = st.number_input('Year', min_value=2010, max_value=2100, step=1)

    # Preprocess user input
    future_data = pd.DataFrame({
        'Store': [store],
        'MarkDown1': [0],
        'MarkDown2': [0],
        'MarkDown3': [0],
        'MarkDown4': [0],
        'MarkDown5': [0],
        'IsHoliday': [0],
        'Dept': [dept],
        'Type': [type],
        'Size': [150000],
        'week': [week],
        'year': [year]
    })

    # Predict sales
    # future_data_processed = preprocess_data(future_data)
    predicted_sales = predict_sales(model, future_data)

    # Display the predicted sales
    st.subheader('Predicted Sales')
    st.write(predicted_sales[0])

if __name__ == '__main__':
    main()