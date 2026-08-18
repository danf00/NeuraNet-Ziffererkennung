import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from save_model import save_model_artifacts

# das mnsit datenset laden und die richtige struktur bringen

# python3 main.py <trainning_data.csv> <output_model.npz>

MODEL_NAME = "feige1.1"

#load data

print(Path(__file__).parent.parent)

df = pd.read_csv(f"{Path(__file__).parent.parent}/archive/mnist_train.csv")

data = df.values

Y_train = data[:, 0]
X_train = data[:, 1:]

X_train = X_train / 255.0  

#weights und bias auf null festlegen

W1 = np.random.randn(784, 128) * np.sqrt(2. / 784)
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 64) * np.sqrt(2. / 128)
b2 = np.zeros((1, 64))

W3 = np.random.randn(64, 10) * np.sqrt(2. / 64)
b3 = np.zeros((1, 10))

# Aktivierungsfunktionen

def ReLu(z):
    a = np.maximum(0, z)
    return a

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shifted)
    exp = exp / np.sum(exp, axis=1, keepdims=True)
    return exp

# Daten durch das System rechnen 

def forwardpass(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = ReLu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = ReLu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def y_onehot(y, num_classes = 10):
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot

onehot = y_onehot(Y_train)

def cross_entry_loss(onehot, A3):
    m = onehot.shape[0]
    epsilon = 1e-12
    loss = -np.sum(onehot * np.log(A3 + epsilon)) / m
    return loss

def relu_ableitung(z):
    return (z > 0).astype(float)

m = X_train.shape[0]

learning_rate = 0.05
epochs = 40000
saved_model = False

for epoch in range(epochs):
    Z1, A1, Z2, A2, Z3, A3 = forwardpass(X_train, W1, b1, W2, b2, W3, b3)

    loss = cross_entry_loss(onehot, A3)

    #delta berechnen#

    delta3 = A3 - onehot
    dW3 = 1/m * (A2.T @ delta3)
    db3 = 1/m * np.sum(delta3, axis=0, keepdims=True)

    delta2 = (delta3 @ W3.T) * relu_ableitung(Z2)  
    dW2 = 1/m * (A1.T @ delta2)
    db2 = 1/m * np.sum(delta2, axis=0, keepdims=True)

    delta1 = (delta2 @ W2.T) * relu_ableitung(Z1)
    dW1 = 1/m * (X_train.T @ delta1)
    db1 = 1/m * np.sum(delta1, axis=0, keepdims=True)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
variant = str(int(epochs / 1000))
variant = variant + "k"

if not saved_model:
    save_model_artifacts("feige1.1", variant, W1, b1, W2, b2, W3, b3, loss, epochs, m)

