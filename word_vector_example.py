"""
Simple example: train a Word2Vec model and print the vector for a given word.
Requires: pip install gensim
"""

from gensim.models import Word2Vec

# Small sample corpus
sentences = [
    ["man", "dog", "animal"],
    ["girl", "pet", "father"],
    ["mother", "cat", "food"],
    ["mango", "burger", "king", "chicken", "rice"],
    ["fries", "rabbit", "son", "prince"],
]

# Train a Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=5000, 
    window=5,
    min_count=1,
    workers=1,
    seed=42,
)
print("#" * 50)
# Ask user for a word and print the vector
word = input("Enter a word to see its vector: ").strip().lower()

if word in model.wv:
    vector = model.wv[word]
    print(f"Vector for '{word}' ({len(vector)} dimensions):")
    print(vector)
    print("---" * 10)

    similar_word = model.wv.most_similar(word)
    print(f"Most similar word to '{word}': {similar_word[0][0]} (similarity: {similar_word[0][1]:.4f})")

else:
    print(f"Word '{word}' was not found in the model vocabulary.")
