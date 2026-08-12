import numpy as np

import pandas as pd


df = pd.read_csv("archive/mnist_test.csv")

data = df.values

modell = np.load("cnn_model_feige1.0.npz")

X_train = data[:, 1:]
Y_train = data[:, 0]

X_train = X_train / 255.0

W1 = modell["W1"]
b1 = modell["b1"]
W2 = modell["W2"]
b2 = modell["b2"]

# Aktivierungsfunktion für nicht linearität

def relu(Z):
    return np.maximum(0, Z)

# softmax funktion für die ausgabe der klasse als Wahrscheinlichkeit

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

random_number = np.random.randint(0, 1000)

Z1, A1, Z2, A2 = forward(X_train[random_number], W1, b1, W2, b2)

predicted_digit = np.argmax(A2, axis=1)[0]
confidence = np.max(A2)

print(f"Echtes Label: {Y_train[random_number]}")
print(f"Vorhersage: {predicted_digit} (Sicherheit: {confidence:.2%})")