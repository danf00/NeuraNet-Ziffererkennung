import pandas as pd
import numpy as np

df = pd.read_csv('archive/mnist_train.csv')

data = df.values

Y_train = data[:, 0]
X_train = data[:, 1:]

X_train = X_train / 255.0

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

def relu(Z):
    return np.maximum(0, Z)

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

Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

print(A2.shape)
print(np.sum(A2, axis=1))
print(A2[0])