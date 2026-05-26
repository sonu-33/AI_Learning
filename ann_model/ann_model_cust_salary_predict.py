## Creating an app where user can provide input and get prediction using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

## Step 1: Load the trained components

model = load_model('salary_model.h5')

with open('scaler_salary.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## Step 2: Create Streamlit app for user input
st.title("Customer Salary Prediction")
st.write("Please provide the following details to predict the customer's salary:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", options=['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100)
tenure = st.slider("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])

## Step 3: Prepare the input data for prediction

## Step 3.1: Create a DataFrame from the user input
input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
}

input_data = pd.DataFrame(input_data)

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])


geo_encoder = onehot_encoder_geo.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([pd.DataFrame(input_data).drop('Geography', axis=1), geo_encoder_df], axis=1)


input_data_scaled = scaler.transform(input_data)

predictions = model.predict(input_data_scaled)

st.success(f"Predicted Estimated Salary: ${predictions[0][0]:,.2f}")

