import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv"
df = pd.read_csv(url)

# Sort data by number of rooms
df = df.sort_values('Avg. Area Number of Rooms')

# Select features (number of rooms) and target (price)
X = df[['Avg. Area Number of Rooms']].values
y = df['Price'].values

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Create sequences for LSTM (e.g., use past 5 values to predict the next)
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split the data into training and testing sets
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1), return_sequences=False),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title('House Price Prediction with LSTM')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.savefig('lstm_price_prediction.png')
plt.close()

# Print the last predicted vs actual price
print(f"Last Actual Price: {y_test[-1][0]:.2f}")
print(f"Last Predicted Price: {y_pred[-1][0]:.2f}")