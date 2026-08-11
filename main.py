import pandas as pd
import numpy as np

# das mnsit datenset laden und die richtige struktur bringen

df = pd.read_csv('archive/mnist_train.csv')

data = df.values

Y_train = data[:, 0]
X_train = data[:, 1:]

X_train = X_train / 255.0

# bias und weights startwerte festlegen

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

def relu(Z):
    """ReLu Aktivierungsfunktion"""
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

def one_hot(y, num_classes = 10):
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot

def cross_entropy_loss(A2, y_onehot):
    m = y_onehot.shape[0]
    loss = -np.sum(y_onehot * np.log(A2)) / m
    return loss

y_onehot = one_hot(Y_train)
loss = cross_entropy_loss(A2, y_onehot)

print(loss)