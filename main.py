from Gen_alg import generate_population
from neural_network import NeuralNetwork

n = 5
structure = generate_population(0, 12, n, "", 3)


def calc_accuracy(structure, num):
    for i in range(num):
        nn = NeuralNetwork(4, 3, 3, structure[i])
        accuracy = nn.calc_accuracy()
        print("Accuracy:", accuracy, structure[i])


calc_accuracy(structure, n)