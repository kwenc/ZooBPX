'''TODO:
    --- człon momentum
    --- wyrkesy 3d
'''
##########################
# autor: Konrad Wenc
# Sieć neuronowa uczona algorytmem wstecznej propagacji z adaptacyjnym wspolczynnikiem uczenia i metoda momentum
###########################
from data_handler import *
from network import *


if __name__ == '__main__':
    training_params, training_labels, test_params, test_labels = prepare_data(sort=True)

    learning_rate = 0.01
    alpha = 0.1

    net = Network([22, 5], learning_rate, alpha)

    net.gradient_descent(training_params, training_labels, test_params, test_labels, epoch=100)
