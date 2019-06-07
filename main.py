'''TODO:
    --- człon momentum
    --- obiektowość
'''
from data_handler import *
from network import *


if __name__ == '__main__':
    # jezeli False dane treningowe sa niesortowane
    training_data, testData = prepare_data(False)

    # podział danych na wejściowe i etykiety
    Pn = training_data[0:15]
    Tn = training_data[16:17][0]

    testPn = testData[0:15]
    testTn = testData[16:17][0]

    lr = 0.01
    Pn = Pn.transpose()
    testPn = testPn.transpose()
    epochNum = 50
    result = neural_network(Pn, Tn, 2, [22, 5], epochNum, lr, testPn, testTn)
