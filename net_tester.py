import network


class NetTester:
    def __init__(self, training_params, training_labels, test_params, test_labels):
        self.training_params = training_params
        self.training_labels = training_labels
        self.test_params = test_params
        self.test_labels = test_labels

    def experiment(self):
        for input_layer in range(10, 30):
            for output_layer in range(5, 30):
                learning_rate = 0.01
                momentum = 0.1
                net = network.Network([input_layer, output_layer], learning_rate, momentum)
                self.gradient_descent(net)

    def gradient_descent(self, net):
        for epoch in range(1, 20000):
            result = []
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
                oe = net.out_error(output, self.training_labels[i])
                sse.append(0.5 * (oe ** 2))
                result.append(output)
                delta_w_b = net.delta(arg, net.weights, net.layers, oe, net.num_layers)
                for k in range(net.num_layers):
                    update = net.weight_update_a(net.weights[k], delta_w_b[k], fe[k], arg[k], net.learning_rate,
                                                 net.biases[k])
                    net.weights[k] = update[0]
                    net.biases[k] = update[1]
                update = net.layer_weight_update(net.weights[net.num_layers], oe, fe[-1], arg[-1],
                                                 net.learning_rate, net.biases[-2])
                net.weights[net.num_layers] = update[0]
                net.biases[-2] = update[1]
                net.biases[-1] += oe

            t_data = net.test_net(net.weights, self.test_params, self.test_labels,
                                  net.layers, net.num_layers, net.biases)

            net.error_plot(t_data, self.test_labels, live=True)

            sum_sse = sum(sse)
            if sum_sse > net.last_cost * net.er:
                net.weights = o_weights
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
        test_result = net.test_net(net.weights, self.test_params, self.test_labels, net.layers,
                                   net.num_layers, net.biases)

        net.error_plot(test_result, self.test_labels, live=False)
        return [test_result[2], test_result[0], net.cost_test, net.ep, net.cost, test_result[1]]
