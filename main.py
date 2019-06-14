'''TODO:
    --- wyrkesy 3d
'''
from data_handler import *
from net_tester import *

if __name__ == '__main__':
    training_params, training_labels, test_params, test_labels = prepare_data(sort=True)

    tester = NetTester(training_params, training_labels, test_params, test_labels)
    tester.experiment()
