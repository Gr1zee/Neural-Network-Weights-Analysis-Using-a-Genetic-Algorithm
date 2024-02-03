import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

for i in range(5):
    nn = NeuralNetwork(4, 3, 3)
    accuracy = nn.calc_accuracy
    print("Accuracy:", nn.calc_accuracy())
    plt.plot(nn.loss_arr)
    plt.show()