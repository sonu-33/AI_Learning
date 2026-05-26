## End to end Deep learning model for RNN models

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping


## Load the IMDB dataset
vocab_size = 10000
dimension = 128

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

## Pad the sequences to a fixed length
maxlen = 300
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

## Build the RNN model
model = Sequential()
model.add(Embedding(vocab_size, dimension, input_length=maxlen)) # Embedding layer to convert word indices to dense vectors of fixed size
model.add(LSTM(dimension, dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

## Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

## Evaluate the model
#model.evaluate(X_test, y_test, batch_size=64)

## Save the model
model.save('rnn_imdb_model.h5')
