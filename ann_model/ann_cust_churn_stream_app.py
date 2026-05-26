## Creating an app where user can provide input and get prediction using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

## Step 1: Load the trained components

model = load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## Step 2: Create Streamlit app for user input
st.title("Customer Churn Prediction")
st.write("Please provide the following details to predict if a customer will churn:")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", options=['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100)
tenure = st.slider("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)


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
    'EstimatedSalary': [estimated_salary]
}

input_data = pd.DataFrame(input_data)

## Step 3.2: Encode categorical variables - Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

## Step 3.3: Encode categorical variables - Geography
geo_encoder = onehot_encoder_geo.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Step 3.4: Combine Geography encoded columns with the original input data
input_data = pd.concat([pd.DataFrame(input_data).drop('Geography', axis=1), geo_encoder_df], axis=1)

## Step 3.5: Normalize the features using the fitted scaler
input_data_scaled = scaler.transform(input_data)

## Step 4: Make predictions using the loaded model and display the result
predictions = model.predict(input_data_scaled)
print ("\\n")
st.write(f"Prediction: {predictions[0][0]}")

if predictions[0][0] >= 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is NOT likely to churn.")


### Below are the steps to use the prediction on Streamlit app:
## On Terminal push the code to Github.
## git add input_app_streamlit.py                                                            
## git commit -m "Committing"
## git push origin main 
## Open browser and go to https://share.streamlit.io/ 
## Open https://ailearning-app.streamlit.app/ -- This is the app created for above code. 
