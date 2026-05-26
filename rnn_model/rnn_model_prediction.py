import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset
vocab_size = 10000
dimension = 128

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

## Pad the sequences to a fixed length
maxlen = 500
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

## Load the saved model
model = load_model('rnn_imdb_model.h5')
model.summary()

## Decode the word indices back to words
def decode_review(encoded_review):
    word_index = imdb.get_word_index()

    #reverse_word_index = {value: key for key, value in word_index.items()}
    reverse_word_index = {}
    for key in word_index:
        value = word_index[key]
        reverse_word_index[value] = key

    #decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    words = []
    for i in encoded_review:
        word = reverse_word_index.get(i - 3, '?')
        words.append(word)
        decoded_review = ' '.join(words)

    return decoded_review

## Preprocess the user input

def preprocess_input(user_input):
    word_index = imdb.get_word_index()
    words = user_input.lower().split() # If the user input has capital letters, it will fail.
    encoded_input = []
    for word in words:
        idx = word_index.get(word, 0) + 3 
        # When Keras built the IMDB dataset, it reserved the first 3 slots for special purposes before assigning any real words. Slot 0 → padding token (used to fill empty spaces in sequences), Slot 1 → start token (marks beginning of a review), Slot 2 → unknown token (for words outside vocab), Slot 3 → "the"(most frequent word), Slot 4 → "a" (2nd most frequent word), Slot 5 → "and"(3rd most frequent word). But imdb.get_word_index() returns a separate lookup table that has no knowledge of those reserved slots. Hence we added +3 to fix words inside vocab
        

        if idx < vocab_size:
            encoded_input.append(idx)
        else: 
            encoded_input.append(2)   # Rare words like "breathtaking" with raw index 61288 is way above 10000 (vocab_size). so it will be replaced with index 2 (which is the index for "unknown" token) to avoid out-of-vocabulary issues.


    padded_input = sequence.pad_sequences([encoded_input], maxlen=maxlen)
    return padded_input

    #User input used: First half was good. Visual grafics were breathtaking. Story, Concept, music were also great. The lead actor could have done better. His expressions were liveless. The 2nd half was bit more dragging. Lot of unnecessary action scenes were added. Thr run time of second half could have reduced. However the ending was good. Except the lead actor, rest all have done good job.  

## Get user input and make a prediction
def predict_sentiment(user_input):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)

    if prediction[0][0] >= 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
  
    return sentiment, prediction[0][0]

## Example usage
if __name__ == "__main__":
    user_input = input("Enter a movie review: ")
    sentiment, confidence = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")

# Data set used are below:
#The first half was good. The visual graphics were breathtaking. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The second half was a bit dragging, and a lot of unnecessary action scenes were added. The runtime of the second half could have been reduced. However, the ending was good. Except for the lead actor, everyone else did a good job.  
#0.12 - Negative

#The first half was good. The visual graphics were great. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The other actors performed extremely well. The second half was a bit dragging, and a lot of unnecessary action scenes were added. The runtime of the second half could have been reduced. However, the ending was good.
#0.17 - Negative

#The first half was good. The visual graphics were great. The story, concept, and music were also great. The lead actor could have done better, as his expressions were lifeless. The other actors performed extremely well. The ending was good.
#0.59 - Positive

#The first half was good. The visual graphics were great. The story, concept, and ending were also good.
#0.70 - Positive
