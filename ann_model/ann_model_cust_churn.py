
# Below code is for trying ANN (Artificial Neural Network) implementation.

## imports for Step 1
import pandas as pd

## imports for Step 2
from sklearn.model_selection import train_test_split # this function splits your dataset into training and testing subsets.
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle # In ML workflows, it's used to save a trained model to a .pkl file so you can reload and reuse it later without retraining 

## imports for Step 3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


## ###################### Step 1: Load the dataset  ###################### 
## Step 1.1: Load the dataset
data = pd.read_csv('Churn_Modelling.csv')
#print (data)


## ###################### Step 2: Preprocess the dataset  ###################### 

## Step 2.1: Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) # axis=1 — tells pandas you're dropping columns (axis=0 would drop rows)
#print (data)


## Step 2.2: Encode categorical variables - Gender (As Gender can be either 0 or 1)
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
#print (data)


## Step 2.3: Encode categorical variables - Geography (As Geography has more than 2 categories, we will use OneHotEncoder)
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
#print (geo_encoder_df)

## Combine Geography encoded columns with the original dataset
data = pd.concat([data.drop('Geography', axis=1),geo_encoder_df], axis=1)
print ("\\n")
print ("1st 5 rows of the preprocessed dataset (RowNumber, CustomerId, Surname, Gender, Geography):"+"\n")
print (data[:5])

## Step 2.4: Save the encoders for future use
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)

with open('onehot_encoder_geo.pkl', 'wb') as f:
    pickle.dump(onehot_encoder_geo, f)

## Step 2.5: Normalizes feature values to a standard range and saves the scaler for future use.

# Step 2.5.1: Divide the dataset to features X (all columns) and target y (Exited column).
X = data.drop('Exited', axis=1) # Features (all columns except 'Exited')
y = data['Exited'] # Target variable (the 'Exited' column)

# Step 2.5.2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test_size=0.2 means 20% of the data will be used for testing, and random_state=42 ensures reproducibility

## Step 2.5.3: Fit StandardScaler on training data and transforms both train and test sets to normalize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit the scaler on the training data and transform it
X_test_scaled = scaler.transform(X_test) # Transform the test data using the same scaler

print ("\\n")
print ("First 5 rows of the scaled training features:"+"\n")
# print (X_train_scaled[:5])
# print ("First 5 rows of the scaled testing features:"+"\n")
# print (X_test_scaled[:5])
# print ("First 5 values of the training target variable:"+"\n")
# print (y_train[:5])
# print ("First 5 values of the testing target variable:"+"\n")
# print (y_test[:5])

## Step 2.5.4: Serializes and saves the fitted scaler to scaler.pkl to be reused on new data later.
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


## ###################### Step 3: Build the ANN model  ###################### 

## #### Step 3.1: Design the model architecture ##########
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), ## Hidden Layer# 1
    Dense(32, activation='relu'), ## Hidden Layer# 2
    Dense(1, activation='sigmoid') ## Outer Layer
    ])

## #### Step 3.2: Compile the model ##########
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


## #### Step 3.3: Train the model ##########
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # This will stop training if the validation loss doesn't improve for 5 consecutive epochs and restore the best weights
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping]) # validation_split=0.2 means 20% of the training data will be used for validation during training

## #### Step 3.4: Plot Training History ##########
# Plot Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')  # saves the chart as an image
plt.show()


## #### Step 3.5: Save the trained model ##########
model.save('churn_model.h5') # This saves the entire model (architecture + weights) to a file named 'churn_model.h5'



