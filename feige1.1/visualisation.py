import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random

df = pd.read_csv(f"{Path(__file__).parent.parent}/archive/mnist_test.csv")
data = df.values

modell = np.load("variants/10k/feige1.1.npz")

W1 = modell["W1"]
b1 = modell["b1"]
W2 = modell["W2"]
b2 = modell["b2"]
W3 = modell["W3"]
b3 = modell["b3"]

def relu(Z):
    return np.maximum(0, Z)

# softmax funktion für die ausgabe der klasse als Wahrscheinlichkeit

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

Y_test = data[:, 0]
X_test = data[:, 1:]

fig, axes = plt.subplots(5, 10, figsize=(15, 8))
fig.suptitle(f"Model: feige1.1--10k-epochs")

for i, ax in enumerate(axes.flat):
    v = random.randint(1,5000)
    Z1, A1, Z2, A2, Z3, A3 = forward(X_test[i], W1, b1, W2, b2, W3, b3)
    pred = np.argmax(A3, axis=1)[0]
    image = X_test[i].reshape(28, 28)
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Pred: {pred}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()

