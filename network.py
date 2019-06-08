import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sigmoid_model import *


class Network:
    def __init__(self, layers, learning_rate, lr_inc=1.05, lr_desc=0.7, er=1.04):
        '''
        Inicializacja wag i biasów Nguyen-Widrow'a
        https://pythonhosted.org/neurolab/index.html
        :param weights:
        :param biases:
        '''
        self.num_layers = len(layers)
        self.layers = layers
        weights = []
        bias = []
        w_fix = 0.7 * (self.layers[0] ** (1 / 15))
        w_rand = (np.random.rand(self.layers[0], 15) * 2 - 1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(self.layers[0], 1)) * w_rand
        w = w_fix * w_rand
        b = np.array([0]) if self.layers[0] == 1 \
            else w_fix * np.linspace(-1, 1, self.layers[0]) * np.sign(w[:, 0])

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

        # dla ostatniej warstwy
        weights.append(np.random.rand(self.layers[-1]))
        bias.append(np.random.rand(1))

        self.weights = weights
        self.biases = bias
        self.learning_rate = learning_rate
        self.lr_inc = lr_inc
        self.lr_desc = lr_desc
        self.er = er
        self.last_cost = 0
        self.cost = []
        self.cost_test = []
        self.ep = 0
        self.goal = 0.0002

    def gradient_descent(self, training_params, training_labels, test_params, test_labels, epoch):
        for j in range(epoch):
            result = []
            # przypisanie wag do zmiennej przed wykonaniem się jednej epoki
            o_weights = self.weights
            # przypisanie wag do zmiennej przed wykonaniem się jednej epoki
            o_bias = self.biases
            sse = []
            for i, inData in enumerate(training_params):
                # tablica przechowująca wektory sygnałów wyjściowych z danych warstw
                fe = []
                # tablica przechowująca wektory łącznych pobudzen neuronów z danych warstw
                arg = []
                # przechowuje listę tablic fe i arg
                fe_arg = []
                fe.append(inData)
                for k in range(self.num_layers):
                    fe_arg = self.hidden_layer(fe[k], self.layers[k], self.weights[k], self.biases[k])
                    fe.append(fe_arg[0])
                    arg.append(fe_arg[1])
                output = self.out_layer(fe[-1], self.weights[-1], self.biases[-1])
                arg.append(sum(fe[-1] * self.weights[-1]))
                oe = self.out_error(output, training_labels[i])
                sse.append(0.5*(oe**2))
                result.append(output)
                delta_w_b = self.delta(arg, self.weights, self.layers, oe, self.num_layers)
                for k in range(self.num_layers):
                    update = self.weight_update_a(self.weights[k], delta_w_b[k], fe[k], arg[k], self.learning_rate, self.biases[k])
                    self.weights[k] = update[0]
                    self.biases[k] = update[1]
                update = self.layer_weight_update(self.weights[self.num_layers], oe, fe[-1], arg[-1],
                                                  self.learning_rate, self.biases[-2])
                self.weights[self.num_layers] = update[0]
                self.biases[-2] = update[1]
                self.biases[-1] += oe

            t_data = self.test_net(self.weights, test_params, test_labels,
                                   self.layers, self.num_layers, self.biases)

            self.error_plot(t_data, test_labels, live=True)

            sum_sse = sum(sse)
            if sum_sse > self.last_cost * self.er:
                self.weights = o_weights
                bias = o_bias
                if self.learning_rate >= 0.0001:
                    self.learning_rate = self.lr_desc * self.learning_rate
            elif sum_sse < self.last_cost:
                learning_rate = self.lr_inc * self.learning_rate
                if learning_rate > 0.99:
                    self.learning_rate = 0.99
            self.last_cost = sum_sse
            self.cost.append(sum_sse)
            self.cost_test.append(t_data[0])
            if t_data[0] < self.goal:
                self.ep = j
                break
            print(f'Epoka #{j:02d} sse: {t_data[0]:.10f}, lr: {self.learning_rate:.4f}, pk: {t_data[2]:.2f}%', end='\r')
            self.ep = j
        test_result = self.test_net(self.weights, test_params, test_labels, self.layers,
                                    self.num_layers, self.biases)

        self.error_plot(test_result, test_labels, live=False)
        return [test_result[2], test_result[0], self.cost_test, self.ep, self.cost, test_result[1]]

    def error_plot(self, test_data, test_labels, live):
        plt.plot(test_data[1], color='#4daf4a', marker='o', label="rozpoznane zwierzeta")
        plt.plot(test_labels, color='#e55964', marker='o', label="oryginalne zwierzeta")
        plt.legend(loc='upper left')
        plt.ylabel('gatunek')
        plt.xlabel('zwierzeta')
        if live:
            plt.draw()
            plt.pause(1e-17)
            plt.clf()
        else:
            plt.show()

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

    def test_net(self, w, test_params, test_labels, neurons_in_layers, layer_num, bias):
        '''Testuje siec na danych testowych'''
        pk = 0
        sse = []
        test_result = []
        for i, tab in enumerate(test_params):
            fe = []
            arg = []
            fe.append(tab)
            for k in range(layer_num):
                fe_arg = self.hidden_layer(fe[k], neurons_in_layers[k], w[k], bias[k])
                fe.append(fe_arg[0])
                arg.append(fe_arg[1])
            y = self.out_layer(fe[-1], w[-1], bias[-1])
            test_result.append(y)
            arg.append(sum(fe[-1] * w[-1]))
            fe.append(y)
            oe = self.out_error(y, test_labels[i])
            if oe ** 2 <= 0.25:
                pk += 1
            sse.append((0.5 * (oe ** 2)))
        pk = pk / (len(test_labels)) * 100
        return [np.sum(np.array(sse)), test_result, pk]

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

    def save_model(self, wages, neurons_in_layers, layer_num, path):
        '''Zapisuje dany model sieci w pliku binarnym'''
        with open(path, 'wb') as f:
            pickle.dump(wages, f)
            pickle.dump(neurons_in_layers, f)
            pickle.dump(layer_num, f)

    def load_model(self, path):
        '''Wczytuje dany model sieci'''
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            neurons_in_layers = pickle.load(f)
            layer_num = pickle.load(f)
        return [weights, neurons_in_layers, layer_num]
