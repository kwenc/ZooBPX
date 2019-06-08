'''TODO:
    --- człon momentum
'''
##########################
# autor: Konrad Wenc
# Sieć neuronowa uczona algorytmem wstecznej propagacji z adaptacyjnym wspolczynnikiem uczenia
###########################
from data_handler import *
from network import *


if __name__ == '__main__':
    # jezeli False dane treningowe sa niesortowane
    training_data, testData = prepare_data(True)

    # podział danych na wejściowe i etykiety
    training_parameters = training_data[0:15]
    training_labels = training_data[16:17][0]

    test_param = testData[0:15]
    test_labels = testData[16:17][0]

    learning_rate = 0.01
    training_parameters = training_parameters.transpose()
    test_param = test_param.transpose()
    epoch_num = 50

    net = Network(training_parameters, training_labels, [22, 5], epoch_num, learning_rate, test_param,
                  test_labels)

    net.learning()
