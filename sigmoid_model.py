import numpy as np


def bipolar_sigmoid(z, bias=1):
    return np.tanh(bias*z)


def bipolar_derivative(z, bias=1):
    return 1 - bipolar_sigmoid(bias * z) ** 2