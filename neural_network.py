import numpy as np
from sklearn import datasets
from Gen_alg import generate_population
import matplotlib.pyplot as plt

a = generate_population(0, 10, 20, "", 3)
INPUT_DIM = 4
OUT_DIM = 3
H_DIM = int(a[1][-3:-1], 2)
print(H_DIM)
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
x = np.random.randn(INPUT_DIM)


def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_fully(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0][y] = 1
    return y_full


def relu_delif(t):
    return (t >= 0).astype(float)


W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

ALPHA = 0.0002
NUM_EPOCHS = int(a[0], 2)
loss_arr = []

print(NUM_EPOCHS)


def neural_network_training(W1, W2, b1, b2):
    for ep in range(NUM_EPOCHS):
        for i in range((len(dataset))):
            # forward
            x, y = dataset[i]
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax(t2)
            E = sparse_cross_entropy(z, y)

            # backward
            y_full = to_fully(y, OUT_DIM)
            dE_dt2 = z - y_full
            dE_dW2 = h1.T @ dE_dt2
            dE_db2 = dE_dt2
            dE_dh1 = dE_dt2 @ W2.T
            dE_dt1 = dE_dh1 * relu_delif(t1)
            dE_dW1 = x.T @ dE_dt1
            dE_db1 = dE_dt1

            W1 = W1 - ALPHA * dE_dW1
            b1 = b1 - ALPHA * dE_db1
            W2 = W2 - ALPHA * dE_dW2
            b2 = b2 - ALPHA * dE_db2

            loss_arr.append(E)
    return W1, W2, b1, b2, loss_arr


W1, W2, b1, b2, loss_arr = neural_network_training(W1, W2, b1, b2)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    return correct / len(dataset)


accuracy = calc_accuracy()
print("Accuracy:", accuracy)
plt.plot(loss_arr)
plt.show()
