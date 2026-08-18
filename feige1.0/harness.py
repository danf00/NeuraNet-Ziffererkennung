import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from save_model import save_model_artifacts


df = pd.read_csv(f"{Path(__file__).parent.parent}/archive/mnist_test.csv")

data = df.values

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

path = Path(__file__).parent
path = path.joinpath("variants")
variants = os.listdir(path)
buffer = []
for item in variants:
    item = item.replace("k", "")
    item = int(item)
    buffer.append(item)

buffer.sort()
variants = buffer

y = []

for x in buffer:
    x = str(x) + "k"
    y.append(x)

variants = y

for variant in variants:

    modell = np.load(f"variants/{variant}/feige1.0.npz")

    X_train = data[:, 1:]
    Y_train = data[:, 0]

    X_train = X_train / 255.0

    W1 = modell["W1"]
    b1 = modell["b1"]
    W2 = modell["W2"]
    b2 = modell["b2"]

    correct = 0
    wrong = 0

    for i in range(9999):
        random = np.random.randint(1, 9999)
        Z1, A1, Z2, A2 = forward(X_train[i], W1, b1, W2, b2)

        if Y_train[i] == np.argmax(A2, axis=1)[0]:
            correct += 1
        else:
            wrong += 1

    print("feige1.0--", variant, ": accuracy: ", correct / (correct + wrong))
