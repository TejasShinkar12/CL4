import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
import networkx as nx
import matplotlib.pyplot as plt

# Load IMDB dataset for sentiment analysis
max_words = 10000
max_len = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Preprocess the data
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Define RNN model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Create a simple network graph to visualize sentiment connections
G = nx.DiGraph()
sentiments = ['Positive', 'Negative']
G.add_nodes_from(sentiments)

# Predict sentiments for a subset of test data
predictions = model.predict(x_test[:100])
for i, pred in enumerate(predictions):
    sentiment = 'Positive' if pred > 0.5 else 'Negative'
    G.add_edge('Review_' + str(i), sentiment)

# Visualize the network graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
plt.title('Network Graph of Sentiment Analysis')
plt.savefig('sentiment_network_graph.png')