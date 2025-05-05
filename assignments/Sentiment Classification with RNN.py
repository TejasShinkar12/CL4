import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split

# Simulated review text data
data = {
    'review': [
        'Great house spacious rooms excellent location',
        'Terrible place noisy dirty small',
        'Lovely property clean modern amenities',
        'Awful experience cramped outdated facilities',
        'Fantastic home beautiful views comfortable',
        'Poor maintenance leaky pipes dark rooms',
        'Wonderful stay perfect cozy welcoming',
        'Bad neighborhood unsafe old house',
        'Amazing value large bright cheerful',
        'Disappointing tiny smelly broken appliances'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
}
df = pd.DataFrame(data)

# Preprocess the text data
max_words = 1000
max_len = 20
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(sequences, maxlen=max_len)
y = df['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define RNN model
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    SimpleRNN(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")