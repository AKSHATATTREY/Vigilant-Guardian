from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Read datasets
neutral_df = pd.read_csv("neutral.txt")
# resting_df = pd.read_csv("resting.txt")
# holding_df = pd.read_csv("holding.txt")
# gripping_df = pd.read_csv("gripping.txt")
punches_df = pd.read_csv("punches.txt")

X = []
y = []
no_of_timesteps = 20

# Function to create sequences
def create_sequences(df, label):
    data = df.iloc[:, 1:].values
    n_samples = len(data)
    for i in range(no_of_timesteps, n_samples):
        X.append(data[i-no_of_timesteps:i, :])
        y.append(label)

# Create sequences for each dataset
create_sequences(neutral_df, 0)
# create_sequences(resting_df, 1)
# create_sequences(holding_df, 2)
# create_sequences(gripping_df, 3)
create_sequences(punches_df, 1)

# Convert lists to numpy arrays and check shapes
X = np.array(X)
y = np.array(y)
print(y)
# Ensure all sequences in X are of the same shape
# The shape of X should be (number_of_samples, no_of_timesteps, number_of_features)
expected_shape = (len(X), no_of_timesteps, X[0].shape[1])
if X.shape[1:] != expected_shape[1:]:
    raise ValueError(f"Inconsistent shape detected: Expected {expected_shape[1:]} but got {X.shape[1:]}")

print(X.shape, y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save("lstm-model.h5")
