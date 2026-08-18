import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from save_model import save_model_artifacts



# das mnsit datenset laden und die richtige struktur bringen

# python3 main.py <trainning_data.csv> <output_model.npz>


MODEL_NAME = "feige1.0"



if len(sys.argv) not in (2, 3):
    print("Usage: python3 main.py <training_data.csv> [output_model_path]")
    sys.exit(1)

training_data_path = sys.argv[1]
output_model_path = sys.argv[2] if len(sys.argv) == 3 else MODEL_NAME


df = pd.read_csv(f"{Path(__file__).parent.parent}/archive/mnist_train.csv")

data = df.values

Y_train = data[:, 0]
X_train = data[:, 1:]

X_train = X_train / 255.0

# bias und weights startwerte festlegen

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

# Aktivierungsfunktion für nicht linearität

def relu(Z):
    return np.maximum(0, Z)

# softmax funktion für die ausgabe der klasse als Wahrscheinlichkeit

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Durchlauf der Daten durch das Netzwerk

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Daten einspeisung


### loss berechnung

# initaliesierung eines arrays für den Vergleich der Vorhersagen mit den tatsächlichen Werten

def one_hot(y, num_classes = 10):
    m = y.shape[0]
    onehot = np.zeros((m, num_classes))
    onehot[np.arange(m), y.astype(int)] = 1
    return onehot



def cross_entropy_loss(A2, y_onehot):
    epsilion = 1e-12
    m = y_onehot.shape[0]
    loss = -np.sum(y_onehot * np.log(A2 + epsilion)) / m
    return loss

y_onehot = one_hot(Y_train)

#### backpropagation

def relu_derivate(z):
    return (z > 0).astype(float)

m = X_train.shape[0]

learning_rate = 0.05
epochs = 3000
saved_model = False


for epoch in range(epochs):

    Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

    loss = cross_entropy_loss(A2, y_onehot)

    delta2 = A2 - y_onehot
    dW2 = 1/m * (A1.T @ delta2)
    db2 = 1/m * np.sum(delta2, axis=0, keepdims=True)

    delta1 = (delta2 @ W2.T) * relu_derivate(Z1)
    dW1 = 1/m * (X_train.T @ delta1)
    db1 = 1/m * np.sum(delta1, axis=0, keepdims=True)

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    #if loss < 0.05:
        #save_model_artifacts(model_dir, W1, b1, W2, b2, float(loss), epoch + 1, m)
        #saved_model = True
        #break
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

variant = str(int(epochs / 1000))
variant = variant + "k"

if not saved_model:
    save_model_artifacts("feige1.0", variant,W1, b1, W2, b2, loss=float(loss), epochs=epochs, image_count=m)

