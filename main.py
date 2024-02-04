import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from Gen_alg import generate_population
from neural_network import NeuralNetwork

structure = generate_population(0, 12, 5, "", 3)
for i in range(5):
    nn = NeuralNetwork(4, 3, 3, structure[i])
    accuracy = nn.calc_accuracy
    print("Accuracy:", nn.calc_accuracy(), structure[i])