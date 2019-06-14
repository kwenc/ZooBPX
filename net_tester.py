import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import network


class NetTester:
    def __init__(self, training_params, training_labels, test_params, test_labels):
        self.training_params = training_params
        self.training_labels = training_labels
        self.test_params = test_params
        self.test_labels = test_labels
        self.net_pk = np.zeros([1])

    def experiment(self):
        in_layer = np.zeros([1])
        out_layer = np.zeros([1])
        for input_layer in range(10, 50):
            for output_layer in range(10, 50):
                in_layer = np.append(in_layer, input_layer)
                out_layer = np.append(out_layer, output_layer)
                net = network.Network([input_layer, output_layer])
                self.net_pk = np.append(self.net_pk, self.gradient_descent(net))
        layers_pk_plot(in_layer, out_layer, self.net_pk)

    def gradient_descent(self, net):
        for epoch in range(1, 100):
            result = list()
            # przypisanie wag do zmiennej przed wykonaniem się jednej epoki
            o_weights = net.weights
            sse = []
            for i, inData in enumerate(self.training_params):
                # tablica przechowująca wektory sygnałów wyjściowych z danych warstw
                fe = []
                # tablica przechowująca wektory łącznych pobudzen neuronów z danych warstw
                arg = []
                # przechowuje listę tablic fe i arg
                fe.append(inData)
                for k in range(net.num_layers):
                    fe_arg = net.hidden_layer(fe[k], net.layers[k], net.weights[k], net.biases[k])
                    fe.append(fe_arg[0])
                    arg.append(fe_arg[1])

                output = net.out_layer(fe[-1], net.weights[-1], net.biases[-1])
                arg.append(sum(fe[-1] * net.weights[-1]))
                out_error = self.training_labels[i] - output
                sse.append(0.5 * (out_error ** 2))
                result.append(output)
                delta_w_b = net.delta(arg, net.weights, net.layers, out_error, net.num_layers)
                for k in range(net.num_layers):
                    update = net.weight_update_a(net.weights[k], net.previous_weights[k], delta_w_b[k], fe[k], arg[k],
                                                 net.learning_rate, net.biases[k])
                    net.weights[k] = update[0]
                    net.previous_weights[k] = update[1]
                    net.biases[k] = update[2]
                update = net.layer_weight_update(net.weights[net.num_layers], net.previous_weights[net.num_layers],
                                                 out_error, fe[-1], arg[-1], net.learning_rate, net.biases[-2])
                net.weights[net.num_layers] = update[0]
                net.previous_weights[net.num_layers] = update[1]
                net.biases[-2] = update[2]
                net.biases[-1] += out_error

            t_data = test_net(net, net.weights, self.test_params, self.test_labels, net.layers, net.num_layers,
                              net.biases)
            # error_plot(t_data, self.test_labels, live=True)
            sum_sse = sum(sse)
            if sum_sse > net.last_cost * net.er:
                net.weights = o_weights
                net.previous_weights = o_weights
                if net.learning_rate >= 0.0001:
                    net.learning_rate = net.lr_dec * net.learning_rate
            elif sum_sse < net.last_cost:
                learning_rate = net.lr_inc * net.learning_rate
                if learning_rate > 0.99:
                    net.learning_rate = 0.99
            net.last_cost = sum_sse
            net.cost.append(sum_sse)
            net.cost_test.append(t_data[0])
            if t_data[0] < net.goal:
                net.ep = epoch
                break
            print(f'Epoka #{epoch:02d} sse: {t_data[0]:.10f}, lr: {net.learning_rate:.4f}, pk: {t_data[2]:.2f}%',
                  end='\r')
            net.ep = epoch
        test_result = test_net(net, net.weights, self.test_params, self.test_labels, net.layers, net.num_layers,
                               net.biases)
        # error_plot(test_result, self.test_labels, live=False)
        # return [pk_array, test_result[0], net.cost_test, net.ep, net.cost, test_result[1]]
        poprawnosc_klasyfikacji = test_result[2]
        return poprawnosc_klasyfikacji


def test_net(net, weight, test_params, test_labels, neurons_in_layers, layer_num, bias):
    '''
    Testuje siec na danych testowych
    :param net:
    :param weight:
    :param test_params:
    :param test_labels:
    :param neurons_in_layers:
    :param layer_num:
    :param bias:
    :return:
    '''
    pk = 0
    sse = []
    test_result = []
    for i, tab in enumerate(test_params):
        fe = []
        arg = []
        fe.append(tab)
        for k in range(layer_num):
            fe_arg = net.hidden_layer(fe[k], neurons_in_layers[k], weight[k], bias[k])
            fe.append(fe_arg[0])
            arg.append(fe_arg[1])
        y = net.out_layer(fe[-1], weight[-1], bias[-1])
        test_result.append(y)
        arg.append(sum(fe[-1] * weight[-1]))
        fe.append(y)
        out_error = test_labels[i] - y
        if out_error ** 2 <= 0.25:
            pk += 1
        sse.append((0.5 * (out_error ** 2)))
    pk = pk / (len(test_labels)) * 100
    return [np.sum(np.array(sse)), test_result, pk]


def error_plot(test_data, test_labels, live):
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


def layers_pk_plot(input_layer, output_layer, pk):
    '''
    Creates 3d plot
    :param input_layer: Array of X axis
    :param output_layer: Array of Y axis
    :param pk: Array of Z axis
    :return: saved plot
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #x, y = np.meshgrid(input_layer, output_layer)

    # Plot the surface.
    surf = ax.plot_trisurf(input_layer, output_layer, pk, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig('pk_warstwy_zaleznosc_posort')
