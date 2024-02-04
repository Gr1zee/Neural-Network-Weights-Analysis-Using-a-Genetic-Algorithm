import numpy as np
from sklearn import datasets
from Gen_alg import generate_population
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, INPUT_DIM, OUT_DIM, H_DIM, structure):
        iris = datasets.load_iris()
        self.INPUT_DIM = INPUT_DIM
        self.OUT_DIM = OUT_DIM
        self.H_DIM = H_DIM

        self.dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

        self.W1 = np.random.rand(self.INPUT_DIM, self.H_DIM)
        self.c = self.structure_change(structure)
        self.b1 = np.random.rand(1, H_DIM)
        self.W2 = np.random.rand(H_DIM, OUT_DIM)
        self.b2 = np.random.rand(1, OUT_DIM)
        self.W1 = self.multiplication(self.W1, self.c)
        self.b1 = self.multiplication(self.b1, self.c)
        self.W2 = self.multiplication(self.W2, self.c)
        self.b2 = self.multiplication(self.b2, self.c)

        self.ALPHA = 0.0002
        self.NUM_EPOCHS = 700
        self.loss_arr = []
        self.W1, self.W2, self.b1, self.b2, self.loss_arr = self.neural_network_training(self.W1, self.W2, self.b1,
                                                                                         self.b2, self.c)

    def relu(self, t):
        return np.maximum(t, 0)

    def structure_change(self, n):
        a = np.zeros((4, 3), int)
        c = 0
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = n[c]
                c += 1
        return a

    def multiplication(self, m1, m2):
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                m1[i][j] = m1[i][j] * m2[i][j]
        return m1

    def softmax(self, t):
        out = np.exp(t)
        return out / np.sum(out)

    def sparse_cross_entropy(self, z, y):
        return -np.log(z[0, y])

    def to_fully(self, y, num_classes):
        y_full = np.zeros((1, num_classes))
        y_full[0][y] = 1
        return y_full

    def relu_delif(self, t):
        return (t >= 0).astype(float)

    def neural_network_training(self, W1, W2, b1, b2, c):
        for ep in range(self.NUM_EPOCHS):
            for i in range((len(self.dataset))):
                # forward
                x, y = self.dataset[i]
                t1 = x @ W1 + b1
                h1 = self.relu(t1)
                t2 = h1 @ W2 + b2
                z = self.softmax(t2)
                E = self.sparse_cross_entropy(z, y)

                # backward
                y_full = self.to_fully(y, self.OUT_DIM)
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = dE_dt2
                dE_dh1 = dE_dt2 @ W2.T
                dE_dt1 = dE_dh1 * self.relu_delif(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = dE_dt1

                W1 = W1 - self.ALPHA * dE_dW1
                b1 = b1 - self.ALPHA * dE_db1
                W2 = W2 - self.ALPHA * dE_dW2
                b2 = b2 - self.ALPHA * dE_db2

                W1 = self.multiplication(W1, c)
                b1 = self.multiplication(b1, c)
                W2 = self.multiplication(W2, c)
                b2 = self.multiplication(b2, c)

                self.loss_arr.append(E)
        return W1, W2, b1, b2, self.loss_arr

    def predict(self, x):
        t1 = x @ self.W1 + self.b1
        h1 = self.relu(t1)
        t2 = h1 @ self.W2 + self.b2
        z = self.softmax(t2)
        return z

    def calc_accuracy(self):
        correct = 0
        for x, y in self.dataset:
            z = self.predict(x)
            y_pred = np.argmax(z)
            if y_pred == y:
                correct += 1
        return correct / len(self.dataset)


if __name__ == "__main__":
    nn = NeuralNetwork(4, 3, 3)
    accuracy = nn.calc_accuracy
    print("Accuracy:", nn.calc_accuracy())
    plt.plot(nn.loss_arr)
    plt.show()
