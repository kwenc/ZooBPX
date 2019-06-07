import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def hidden_layer(tab, num_of_neurons, weight, bias):
    """obliczanie łącznego pobudzenia neuronów w warstwie oraz sygnałów wyjściowych z neuronów"""
    arg = []
    ee = []
    for i in range(num_of_neurons):
        e = sum(tab * weight[i]) + bias[i]
        ee.append(e)
        arg.append(bipolar_sigmoid(e))
    return [arg, ee]


def out_layer(tab, weight, bias):
    '''Obliczanie sygnału wyjściowego w ostatniej warstwie'''
    tab = np.asarray(tab)
    x = sum(tab*weight) + bias[0]
    return x


def out_error(out, val):
    return val - out


def error_l(l, weight, fe):
    '''Obliczanie delty dla ostatniej warstwy'''
    err = []
    for k, val in enumerate(fe):
        err.append(l * weight[k] * bipolar_derivative(val))
    return err


def error_a(err, weight, fe):
    '''Obliczanie delty dla pozostałych warstw, procz ostatniej'''
    err = np.asarray(err)
    x = bipolar_derivative(fe) * sum(weight * err)
    return x


def weight_update_a(weight, errors, arg, fe, learning_rate, bias):
    '''Aktualizacja wag dla wszystkich warstw procz ostatniej'''
    for i, val in enumerate(weight):
        bias[i] += learning_rate * errors[i]
        for j in range(val.size):
            weight[i][j] += learning_rate * errors[i] * arg[j]
    return [weight, bias]


def layer_weight_update(weight, oe, arg, out, learning_rate, bias):
    '''Aktualizacja wag między ostatnią warstwą a ostatnią ukrytą warstwą'''
    for i in range(weight.size):
        bias[i] += learning_rate * oe * 1
        weight[i] += learning_rate * oe * 1 * arg[i]
    return [weight, bias]


def save_model(wages, neurons_in_layers, layer_num, path):
    '''Zapisuje dany model sieci w pliku binarnym'''
    with open(path, 'wb') as f:
        pickle.dump(wages, f)
        pickle.dump(neurons_in_layers, f)
        pickle.dump(layer_num, f)


def load_model(path):
    '''Wczytuje dany model sieci'''
    weights = []
    neurons_in_layers = []
    with open(path, 'rb') as f:
        weights = pickle.load(f)
        neurons_in_layers = pickle.load(f)
        layer_num = pickle.load(f)
    return [weights, neurons_in_layers, layer_num]


def test_net(w, test_Pn, test_Tn, neurons_in_layers, layer_num, bias):
    '''Testuje siec na danych testowych'''
    pk = 0
    sse = []
    test_result = []
    for i, tab in enumerate(test_Pn):
        fe = []
        arg = []
        fe_arg = []
        fe.append(tab)
        for k in range(layer_num):
            fe_arg = hidden_layer(fe[k], neurons_in_layers[k], w[k], bias[k])
            fe.append(fe_arg[0])
            arg.append(fe_arg[1])
        y = out_layer(fe[-1], w[-1], bias[-1])
        test_result.append(y)
        arg.append(sum(fe[-1] * w[-1]))
        fe.append(y)
        oe = out_error(y, test_Tn[i])
        if oe**2 <= 0.25:
            pk += 1
        sse.append((0.5*(oe**2)))
    pk = pk / (len(test_Tn)) * 100
    return [np.sum(np.array(sse)), test_result, pk]


def init_NW(neurons_in_layers, layer_num):
    '''
    Inicializacja wag i biasów Nguyen-Widrow'a

    funkja wzorowana na funkcji z bliblioteki NeuroLab
    https://pythonhosted.org/neurolab/index.html
    '''
    weights = []
    bias = []
    w_fix = 0.7 * (neurons_in_layers[0] ** (1 / 15))
    w_rand = (np.random.rand(neurons_in_layers[0], 15) * 2 - 1)
    w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neurons_in_layers[0], 1)) * w_rand
    w = w_fix*w_rand
    b = np.array([0]) if neurons_in_layers[0] == 1\
        else w_fix * np.linspace(-1, 1, neurons_in_layers[0]) * np.sign(w[:, 0])

    weights.append(w)
    bias.append(b)
    for i in range(1, layer_num):
        w_fix = 0.7 * (neurons_in_layers[i] ** (1 / neurons_in_layers[i - 1]))
        w_rand = (np.random.rand(neurons_in_layers[i], neurons_in_layers[i - 1]) * 2 - 1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neurons_in_layers[i], 1)) * w_rand
        w = w_fix*w_rand
        b = np.array([0]) if neurons_in_layers[i] == 1\
            else w_fix * np.linspace(-1, 1, neurons_in_layers[i]) * np.sign(w[:, 0])
        weights.append(w)
        bias.append(b)
    # dla ostatniej warstwy
    weights.append(np.random.rand(neurons_in_layers[-1]))
    bias.append(np.random.rand(1))
    return [weights, bias]


def delta(arg, weights, neurons_in_layers, oe, layer_num):
    """Oblicza delte przy propagacji wstecznej dla wszystkich warstw"""
    # odwrócona tablica wektorów łącznych pobudzen neuronów z danych warstw
    der_Fe = arg[::-1]
    # odwrócona tablica wag
    wage_fl = weights[::-1]
    # odwrócona tablica z iloscia neuronów w danej warstwie
    nil_fl = neurons_in_layers[::-1]
    # tablica przechowująca wektory błędów dla danej warstwy
    d = []
    d.append(error_l(oe, wage_fl[0], der_Fe[1]))
    for k in range(1, layer_num):
        temp = wage_fl[k]
        temp = temp.transpose()
        temp_d = []
        dfe = der_Fe[k+1]
        for p in range(nil_fl[k]):
            temp_d.append(error_a(d[k-1], temp[p], dfe[p]))
        d.append(np.asarray(temp_d))
    # odwrócenie tablicy błędów
    d = d[::-1]
    return d


def neural_network(Pn, Tn, layer_num, neurons_in_layers, epoch_num, learning_rate, test_Pn, test_Tn,
                   lr_inc=1.05, lr_desc=0.7, er=1.04):
    """Główna funkcja odpowiadająca za sieć neuronową
        weights, bias - zwracana tablica przechowujaca wektory wagowe i przesuniecia
        last_cost - wartosc funkcji kosztu w poprzedniej chwili czasu
        result - tablica przechowująca wyjścia sieci dla danej epoki
    """
    cost = []
    cost_test = []
    ep = 0
    goal = 0.0002
    weights, bias = init_NW(neurons_in_layers, layer_num)
    last_cost = 0
    for j in range(epoch_num):
        result = []
        # przypisanie wag do zmiennej przed wykonaniem się jednej epoki
        o_weights = weights
        # przypisanie wag do zmiennej przed wykonaniem się jednej epoki
        o_bias = bias
        sse = []
        for i, inData in enumerate(Pn):
            # tablica przechowująca wektory sygnałów wyjściowych z danych warstw
            fe = []
            # tablica przechowująca wektory łącznych pobudzen neuronów z danych warstw
            arg = []
            # przechowuje listę tablic fe i arg
            fe_arg = []
            fe.append(inData)
            for k in range(layer_num):
                fe_arg = hidden_layer(fe[k], neurons_in_layers[k], weights[k], bias[k])
                fe.append(fe_arg[0])
                arg.append(fe_arg[1])
            output = out_layer(fe[-1], weights[-1], bias[-1])
            arg.append(sum(fe[-1] * weights[-1]))
            oe = out_error(output, Tn[i])
            sse.append(0.5*(oe**2))
            result.append(output)
            delta_w_b = delta(arg, weights, neurons_in_layers, oe, layer_num)
            for k in range(layer_num):
                update = weight_update_a(weights[k], delta_w_b[k], fe[k], arg[k], learning_rate, bias[k])
                weights[k] = update[0]
                bias[k] = update[1]
            update = layer_weight_update(weights[layer_num], oe, fe[-1], arg[-1], learning_rate, bias[-2])
            weights[layer_num] = update[0]
            bias[-2] = update[1]
            bias[-1] += oe

        t_data = test_net(weights, test_Pn, test_Tn, neurons_in_layers, layer_num, bias)

        ######### live plot #########
        plt.plot(t_data[1], color='#4daf4a', marker='o', label="wyjscie sieci")
        plt.plot(test_Tn, color='#e55964', marker='o', label="target")
        plt.legend(loc='upper left')
        plt.ylabel('klasa')
        plt.xlabel('wzorzec')
        plt.draw()
        plt.pause(1e-17)
        plt.clf()
        #############################

        sum_sse = sum(sse)
        if sum_sse > last_cost*er:
            weights = o_weights
            bias = o_bias
            if learning_rate >= 0.0001:
                learning_rate = lr_desc * learning_rate
        elif sum_sse < last_cost:
            learning_rate = lr_inc * learning_rate
            if learning_rate > 0.99:
                learning_rate = 0.99
        last_cost = sum_sse
        cost.append(sum_sse)
        cost_test.append(t_data[0])
        if t_data[0] < goal:
            ep = j
            break
        print(f'Epoka #{j:02d} sse: {t_data[0]:.10f}, lr: {learning_rate:.4f}, pk: {t_data[2]:.2f}%', end='\r')
        ep = j
    test_result = test_net(weights, test_Pn, test_Tn, neurons_in_layers, layer_num, bias)

    plt.plot(test_result[1], color='#4daf4a', marker='o', label="wyjscie sieci")
    plt.plot(test_Tn, color='#e55964', marker='o', label="target")
    plt.legend(loc='upper left')
    plt.ylabel('klasa')
    plt.xlabel('wzorzec')
    plt.show()
    return [test_result[2], test_result[0], cost_test, ep, cost, test_result[1]]


def bipolar_sigmoid(z, bias=1):
    '''funkcja aktywacji neuronu: sigmoid bipolarny'''
    return np.tanh(bias*z)


def bipolar_derivative(z, bias=1):
    return 1 - bipolar_sigmoid(bias * z) ** 2
