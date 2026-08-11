import pandas as pd
import numpy as np

df = pd.read_csv('archive/mnist_train.csv')

data = df.values

Y_train = data[:, 0]
X_train = data[:, 1:]

X_train = X_train / 255.0

