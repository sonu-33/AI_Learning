### One hot representation
"""
from tensorflow.keras.preprocessing.text import one_hot

## Import dataset

sentences = [
'the cat sat on the mat', 
'the cat sat on the log',
'the dog sat on the log', 
'the cat and the dog are friends',
'the dog and the cat are enemies',
'the cat is on the roof']

#[[516, 2056, 7599, 7198, 516, 8403], 
#[516, 2056, 7599, 7198, 516, 6126], 
#[516, 6047, 7599, 7198, 516, 6126], 
#[516, 2056, 1345, 516, 6047, 968, 3571], 
#[516, 6047, 1345, 516, 2056, 968, 4483], 
#[516, 2056, 7246, 7198, 516, 7193]]

#[[3029, 9837, 6037, 2976, 3029, 4159], 
#[3029, 9837, 6037, 2976, 3029, 7320], 
#[3029, 442, 6037, 2976, 3029, 7320], 
#[3029, 9837, 8601, 3029, 442, 8592, 8184], 
#[3029, 442, 8601, 3029, 9837, 8592, 3780], 
#[3029, 9837, 4873, 2976, 3029, 874]]


## Vocabulary size
vocab_size = 10000

## One hot representation
onehot_repr = [one_hot(words, vocab_size) for words in sentences]

print(onehot_repr)
"""

### Embedding Representation

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential

import numpy as np

## Import dataset
sentences = [
'the cat sat on the mat', 
'the cat sat on the log',
'the dog sat on the log', 
'the cat and the dog are friends',
'the dog and the cat are enemies',
'the cat is on the roof']

## Vocabulary size
vocab_size = 10000

## One hot representation
onehot_repr = [one_hot(words, vocab_size) for words in sentences]

## Embedding representation
sentences_length = 10
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sentences_length)
#embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=sentences_length)

#print(embedded_docs)

## Create model

dim = 8 #dimension of the dense embedding
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=sentences_length))
model.compile('adam', 'mse')
model.build(input_shape=(None, sentences_length))
model.summary()

print(model.predict(embedded_docs)) # embedded_docs (after pad sequences added) goes through Embedding layer and returns actual dense vectors.
