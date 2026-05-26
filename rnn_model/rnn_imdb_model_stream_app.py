## Creating an app where user can provide input and get prediction using Streamlit

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


## Step 1: Load the IMDB dataset
vocab_size = 10000
dimension = 128

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

## Step 1.1: Pad the sequences to a fixed length
maxlen = 500
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

## Step 1.2: Load the trained components
model = load_model('rnn_imdb_model.h5')

## Step 2: Create Streamlit app for user input
st.title("IMDB Sentiment Analysis")
st.write("Please provide a movie review to analyze its sentiment:")
user_review = st.text_area("Movie Review", height=150)

## Step 3: Preprocess the user input

## Step 3.1: Decode the word indices back to words
def decode_review(encoded_review):
    word_index = imdb.get_word_index()

    reverse_word_index = {}
    for key in word_index:
        value = word_index[key]
        reverse_word_index[value] = key

    words = []
    for i in encoded_review:
        word = reverse_word_index.get(i - 3, '?')
        words.append(word)
        decoded_review = ' '.join(words)

    return decoded_review

## Step 3.2: Preprocess the user input
def preprocess_input(user_input):
    word_index = imdb.get_word_index()
    words = user_input.lower().split() 
    encoded_input = []
    for word in words:
        idx = word_index.get(word, 0) + 3 
       
        if idx < vocab_size:
            encoded_input.append(idx)
        else: 
            encoded_input.append(2)   

    padded_input = sequence.pad_sequences([encoded_input], maxlen=maxlen)
    return padded_input


## Step 3.3: Get user input and make a prediction
def predict_sentiment(user_input):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)

    if prediction[0][0] >= 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
  
    return sentiment, prediction[0][0]

## Step 4: Make prediction and display results
user_input = user_review.strip()
sentiment, confidence = predict_sentiment(user_input)
st.write(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")

if confidence >= 0.5:
    st.write("The review is likely positive.")
else:
    st.write("The review is likely negative.")

# Data set used are below:
#The first half was good. The visual graphics were breathtaking. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The second half was a bit dragging, and a lot of unnecessary action scenes were added. The runtime of the second half could have been reduced. However, the ending was good. Except for the lead actor, everyone else did a good job.  
#0.12 - Negative

#The first half was good. The visual graphics were great. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The other actors performed extremely well. The second half was a bit dragging, and a lot of unnecessary action scenes were added. The runtime of the second half could have been reduced. However, the ending was good.
#0.17 - Negative

#The first half was good. The visual graphics were great. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The other actors performed extremely well. The ending was good.
#0.59 - Positive

#The first half was good. The visual graphics were great. The story, concept, and ending were also good.
#0.70 - Positive

