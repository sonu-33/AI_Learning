### Determining the optimal number of hidden layers and neurons for an Artificial Neural Network (ANN) 
#This can be challenging and often requires experimentation. However, there are some guidelines and methods that can help you in making an informed decision:

# - Start Simple: Begin with a simple architecture and gradually increase complexity if needed.
# - Grid Search/Random Search: Use grid search or random search to try different architectures.
# - Cross-Validation: Use cross-validation to evaluate the performance of different architectures.
# - Heuristics and Rules of Thumb: Some heuristics and empirical rules can provide starting points, such as:
#   -    The number of neurons in the hidden layer should be between the size of the input layer and the size of the output layer.
#   -  A common practice is to start with 1-2 hidden layers.


import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

## ###################### Step 1: Load the dataset  ###################### 
## Step 1.1: Load the dataset
data = pd.read_csv('Churn_Modelling.csv')
#print (data)


## ###################### Step 2: Preprocess the dataset  ###################### 

## Step 2.1: Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) # axis=1 — tells pandas you're dropping columns (axis=0 would drop rows)


## Step 2.2: Encode categorical variables - Gender (As Gender can be either 0 or 1)
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])


## Step 2.3: Encode categorical variables - Geography (As Geography has more than 2 categories, we will use OneHotEncoder)
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine Geography encoded columns with the original dataset
data = pd.concat([data.drop('Geography', axis=1),geo_encoder_df], axis=1)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

## Step 2.5.3: Fit StandardScaler on training data and transforms both train and test sets to normalize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

## Step 2.5.4: Serializes and saves the fitted scaler to scaler.pkl to be reused on new data later.
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


## #### Step 3: Create the model ##########
def create_model(neurons=16, layers=1, epochs=100, batch_size=32):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    #return model

    for i in range(layers-1):
        model.add(Dense(neurons,activation='relu'))
    #return model

    model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

## #### Step 4: Create a Keras Classifier ##########
keras_classifier = KerasClassifier(model=create_model, epochs=100, batch_size=32)


## #### Step 5: Define grid search parameters ##########
param_grid = {
    'model__neurons': [16, 32, 64], # Number of neurons in the hidden layer
    'model__epochs': [50, 100],
    'model__batch_size': [16, 32],
    'model__layers': [1, 2, 3]
} 

## #### Step 6: Perform Grid Search with Cross-Validation ##########
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_result = grid_search.fit(X_train_scaled, y_train)

## #### Step 7: Print the best parameters and best score ##########
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

#### Result of the above: Best: 0.85725 using {'model__batch_size': 16, 'model__epochs': 50, 'model__layers': 1, 'model__neurons': 16}

### ann_model_cust_churn_predict.py before applying above optimal parameters: 0.733677
### ann_model_cust_churn_predict.py after applying above optimal parameters: 0.7707571