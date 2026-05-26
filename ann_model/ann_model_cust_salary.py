## Below code is to create ANN model for predicting customer salary.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#################################################################################################

## Step 1: Load the dataset and Preprocess the dataset
data = pd.read_csv("Churn_Modelling.csv")

data = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

data = pd.concat([data.drop('Geography', axis=1),geo_encoder_df], axis=1)

#################################################################################################

## Step 2: Create a scaler for future use.

# Step 2.1: Target variable is 'EstimatedSalary' column
X = data.drop('EstimatedSalary', axis=1) 
y = data['EstimatedSalary'] 

# Step 2.2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

with open('scaler_salary.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#################################################################################################

## #### Step 3: Design the model ##########

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), ## Hidden Layer# 1
    Dense(32, activation='relu'), ## Hidden Layer# 2
    Dense(1, activation='linear')## Outer Layer
    ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping]) 

model.save('salary_model.h5') 

#################################################################################################
