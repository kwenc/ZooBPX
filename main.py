'''TODO:
    --- człon momentum
'''
##########################
# autor: Konrad Wenc
# Sieć neuronowa uczona algorytmem wstecznej propagacji z adaptacyjnym wspolczynnikiem uczenia i metoda momentum
###########################
from data_handler import *
from network import *


if __name__ == '__main__':
    # jezeli False dane treningowe sa niesortowane
    training_data, testData = prepare_data(True)

    # podział danych na wejściowe i etykiety
    training_parameters = training_data[0:15]
    training_labels = training_data[16:17][0]

    test_params = testData[0:15]
    test_labels = testData[16:17][0]

    learning_rate = 0.01
    training_parameters = training_parameters.transpose()
    test_params = test_params.transpose()

    net = Network([22, 5], learning_rate)

    net.gradient_descent(training_parameters, training_labels, test_params, test_labels, epoch=100)
