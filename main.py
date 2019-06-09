'''TODO:
    --- człon momentum
    --- wyrkesy 3d
'''
##########################
# autor: Konrad Wenc
# Sieć neuronowa uczona algorytmem wstecznej propagacji z adaptacyjnym wspolczynnikiem uczenia i metoda momentum
###########################
from data_handler import *
from net_tester import *

if __name__ == '__main__':
    training_params, training_labels, test_params, test_labels = prepare_data(sort=True)

    tester = NetTester(training_params, training_labels, test_params, test_labels)
    tester.experiment()
