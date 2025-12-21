import numpy as np
import random
import sys
import gzip
import pickle


def sigmoid(product):
    return 1 / (1 + np.exp(-product))

def cost_function(net, num_set):
    length = len(num_set[0])
    answer = 0

    for i in range(length):
        desired_number = num_set[1][i]
        number_set = num_set[0][i]

        y = np.zeros((10 ,1), dtype=int)
        y[desired_number] = 1

        number_set = number_set.reshape(784, 1)
        a = net.feedforward(number_set)

        answer += np.power((y - a), 2)
        
    return np.sum(answer / (2*length))



class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.a_layer = []
        self.weight = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
    
    def feedforward(self, input):
        self.a_layer = []
        a = input
        for w, b in zip(self.weight, self.bias):
            self.a_layer.append(a)
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, set_input, eta, batch_size, num_epoch):
        for i in range(num_epoch):
            X = set_input[0]   # first np.array
            y = set_input[1]   # second np.array

            # generate a permutation of indices
            perm = np.random.permutation(len(X))

            # Apply same permutation to both arrays
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            shuffled_data = (X_shuffled, y_shuffled)

            mini_batches = []

            for k in range(0, len(X_shuffled), batch_size):
                images = shuffled_data[0]
                labels = shuffled_data[1]
                data = (images[k:k+batch_size], labels[k:k+batch_size])

                mini_batches.append(data)
                    
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            val = cost_function(self, shuffled_data)
            print(f"Total cost in epoch {i+1}: {val}")

    def update_mini_batch(self, mini_batch, eta):
        final_weight = [np.zeros(w.shape) for w in self.weight]
        final_bias = [np.zeros(b.shape) for b in self.bias]

        for i in range(len(mini_batch[0])):
            number_set = mini_batch[0][i]
            desired_number = mini_batch[1][i]

            number_set = number_set.reshape(784, 1)

            y = np.zeros((10 ,1), dtype=int)
            y[desired_number] = 1

            a = self.feedforward(number_set)

            errorL = (a - y) * (a * (1 - a))

            error = self.backpropagation(errorL)

            bias_gradient = error

            weight_gradient = []

            for l in range(self.num_layers - 1):
                weight_gradient.append(np.dot(error[l], self.a_layer[l].T))

            for l in range(self.num_layers - 1):
                final_weight[l] += weight_gradient[l]
                final_bias[l] += bias_gradient[l]
            
        avg_bias_gradient = [b / len(mini_batch[1]) for b in final_bias]
        avg_weight_gradient = [w / len(mini_batch[1]) for w in final_weight]

        for l in range(len(self.weight)):
            self.weight[l] -= eta * avg_weight_gradient[l]
            self.bias[l]  -= eta * avg_bias_gradient[l]

    def backpropagation(self, errorL):
        errors = [errorL]
        error = errorL

        layer = self.num_layers - 2

        while layer != 0:
            previos_layer = self.sizes[layer]
            this_layer = self.sizes[layer+1]
            error = np.dot(self.weight[layer].T, error) * (self.a_layer[layer] * (1 - self.a_layer[layer]))
            errors.insert(0, error)
            layer -= 1
        
        return errors

if __name__ == "__main__":
    # Creation of network with 4 layers, first one will have 784 neurons
    # second one will have 15 neurons and third one will also have 15 neurons
    # last one will have 10 neurons, it will represent digits from 0 to 9
    net = Network([784, 15, 15, 10])    
    with gzip.open("/Users/azizsattarov/Desktop/Federated Learning/neural-networks-and-deep-learning/data/mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    net.SGD(train_set, 3.0, 10, 30)

    amount_net = len(test_set[1])
    correct = 0
    for i in range(amount_net):
        imageValue = np.array(test_set[0][i])

        imageValue = imageValue.reshape(784, 1)    

        net_output = net.feedforward(imageValue)

        desired_output = test_set[1][i]

        predicted_digit = np.argmax(net_output)

        if predicted_digit == desired_output:
            correct += 1
    
    print(f"Accuracy = {correct/amount_net}")