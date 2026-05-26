## This script loads the trained ANN model and its associated preprocessing components (scaler and one-hot encoder) to make predictions on new data. It demonstrates how to preprocess new input data in the same way as the training data, ensuring that the model can make accurate predictions.

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

## Step 1: Load the trained components

model = load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## Step 2: Prepare new input data
# Example new data (replace with actual new data)
input_data = {
    'CreditScore': [300],
    'Geography': ['France'],
    'Gender': ['Male'],
    'Age': [35],
    'Tenure': [5],
    'Balance': [100000],
    'NumOfProducts': [3],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [5000]
}

## Step 3: Preprocess the new input data
input_data = pd.DataFrame(input_data)

## Step 3.1: Encode categorical variables - Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

## Step 3.2: Encode categorical variables - Geography
geo_encoder = onehot_encoder_geo.transform(input_data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Step 3.3: Combine Geography encoded columns with the original input data
input_data = pd.concat([pd.DataFrame(input_data).drop('Geography', axis=1), geo_encoder_df], axis=1)

print ("\\n")
print ("Preprocessed input data:"+"\n")
print (input_data)

## Step 3.4: Normalize the features using the fitted scaler
input_data_scaled = scaler.transform(input_data)

print ("\\n")
print ("Scaled input data:"+"\n")
print (input_data_scaled)

## Step 4: Make predictions using the loaded model
predictions = model.predict(input_data_scaled)

#print ("\\n")
#print ("Predictions: "+ str(predictions))
print ("\\n")
print ("Predictions: "+ str(predictions[0][0]))

if predictions[0][0] >= 0.5:
    print ("The model predicts that the customer will churn (Exited=1).")
else:
    print ("The model predicts that the customer will not churn (Exited=0).")
