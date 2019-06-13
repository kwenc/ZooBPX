import matplotlib.pyplot as plt
from sigmoid_model import *


class Network:
    def __init__(self, layers, learning_rate, momentum):
        '''
        Inicializacja wag i biasów Nguyen-Widrow'a
        '''
        self.num_layers = len(layers)
        self.layers = layers
        weights = []
        bias = []
        w_fix = 0.7 * (self.layers[0] ** (1 / 15))
        w_rand = (np.random.rand(self.layers[0], 15) * 2 - 1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(self.layers[0], 1)) * w_rand
        w = w_fix * w_rand
        b = np.array([0]) if self.layers[0] == 1 else w_fix * np.linspace(-1, 1, self.layers[0]) * np.sign(w[:, 0])

        weights.append(w)
        bias.append(b)
        for i in range(1, self.num_layers):
            w_fix = 0.7 * (self.layers[i] ** (1 / self.layers[i - 1]))
            w_rand = (np.random.rand(self.layers[i], self.layers[i - 1]) * 2 - 1)
            w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(self.layers[i], 1)) * w_rand
            w = w_fix * w_rand
            b = np.array([0]) if self.layers[i] == 1 \
                else w_fix * np.linspace(-1, 1, self.layers[i]) * np.sign(w[:, 0])
            weights.append(w)
            bias.append(b)

        # Inicjalizacja wag i biasow dla ostatniej warstwy
        weights.append(np.random.rand(self.layers[-1]))
        bias.append(np.random.rand(1))

        self.weights = weights
        self.biases = bias
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lr_inc = 1.05
        self.lr_dec = 0.7
        self.er = 1.04
        self.last_cost = 0
        self.cost = []
        self.cost_test = []
        self.ep = 0
        self.goal = 0.0002

    def hidden_layer(self, tab, num_of_neurons, weight, bias):
        """obliczanie łącznego pobudzenia neuronów w warstwie oraz sygnałów wyjściowych z neuronów"""
        arg = []
        ee = []
        for i in range(num_of_neurons):
            e = sum(tab * weight[i]) + bias[i]
            ee.append(e)
            arg.append(bipolar_sigmoid(e))
        return [arg, ee]

    def out_layer(self, tab, weight, bias):
        '''Obliczanie sygnału wyjściowego w ostatniej warstwie'''
        tab = np.asarray(tab)
        x = sum(tab * weight) + bias[0]
        return x

    def out_error(self, out, val):
        return val - out

    def error_l(self, l, weight, fe):
        '''Obliczanie delty dla ostatniej warstwy'''
        err = []
        for k, val in enumerate(fe):
            err.append(l * weight[k] * bipolar_derivative(val))
        return err

    def error_a(self, err, weight, fe):
        '''Obliczanie delty dla pozostałych warstw, procz ostatniej'''
        err = np.asarray(err)
        x = bipolar_derivative(fe) * sum(weight * err)
        return x

    def weight_update_a(self, weight, errors, arg, fe, learning_rate, bias):
        '''Aktualizacja wag dla wszystkich warstw procz ostatniej'''
        for i, val in enumerate(weight):
            bias[i] += learning_rate * errors[i]
            for j in range(val.size):
                weight[i][j] += learning_rate * errors[i] * arg[j]
        return [weight, bias]

    def layer_weight_update(self, weight, oe, arg, out, learning_rate, bias):
        '''Aktualizacja wag między ostatnią warstwą a ostatnią ukrytą warstwą'''
        for i in range(weight.size):
            bias[i] += learning_rate * oe * 1
            weight[i] += learning_rate * oe * 1 * arg[i]
        return [weight, bias]

    def delta(self, arg, weights, neurons_in_layers, oe, layer_num):
        """Oblicza delte przy propagacji wstecznej dla wszystkich warstw"""
        # odwrócona tablica wektorów łącznych pobudzen neuronów z danych warstw
        der_Fe = arg[::-1]
        # odwrócona tablica wag
        wage_fl = weights[::-1]
        # odwrócona tablica z iloscia neuronów w danej warstwie
        nil_fl = neurons_in_layers[::-1]
        # tablica przechowująca wektory błędów dla danej warstwy
        d = []
        d.append(self.error_l(oe, wage_fl[0], der_Fe[1]))
        for k in range(1, layer_num):
            temp = wage_fl[k]
            temp = temp.transpose()
            temp_d = []
            dfe = der_Fe[k + 1]
            for p in range(nil_fl[k]):
                temp_d.append(self.error_a(d[k - 1], temp[p], dfe[p]))
            d.append(np.asarray(temp_d))
        # odwrócenie tablicy błędów
        d = d[::-1]
        return d
