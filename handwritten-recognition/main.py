import numpy as np
import random
import sys
import gzip
import pickle



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def cost_function_avg(net_value):
    amount_net = len(net_value[1])

    total_cost = 0

    for i in range(amount_net):
        imageValue = np.array(net_value[0][i])

        imageValue = imageValue.reshape(784, 1)    

        # print(f"Value {i}: {imageValue.shape}")

        net_output = net.feedforward(imageValue)

        # print(f"Actual number {i}: {net_value[1][i]}")

        desired_output = np.zeros((10, 1), dtype=int)

        desired_output[net_value[1][i]] = 1

        # print(f"Desired output: {desired_output}")

        cost_result = pow((desired_output - net_output),2)

        total_cost += cost_result

    return (total_cost/(amount_net * 2)).sum()

        

# Representation of neural network, it initialization
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layer = []
        #Bias and weight are initialised randomly, but there is a better way to do it
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        # print(f"Bias: {self.biases}")
        # print(f"Weight: {self.weights}")
        self.layer = []
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print(f"\nLayer {i+1}:")
            # print(f"Bias in loop: {b}")
            # print(f"Weight in loop: {w}")
            self.layer.append(a)
            a = sigmoid(np.dot(w, a)+b)
            # print(f"a: {a}")
        return a
    
    def SGD(self, data_set, batch_size):
        """Stochastic gradient descent learning"""
        """First of all we divide our data into mini-batches"""
        """Second of all"""

        X = data_set[0]   # first np.array
        y = data_set[1]   # second np.array

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
            self.update_mini_batch(mini_batch)
    
    def update_mini_batch(self, batch):
        for i in range(len(batch[1])):
            value = np.array(batch[0][i])

            value = value.reshape(784, 1)    

            net_output = self.feedforward(value)

            desired_output = np.zeros((10, 1), dtype=int)

            desired_output[batch[1][i]] = 1

            delta = self.backpropogation(net_output, desired_output)

    def backpropogation(self, net_output, desired_output):
        # Computed output layer
        delta = (net_output - desired_output) * (net_output * (1-net_output))

        # Computed hidden layers
        i = self.num_layers-2
        while i != 0:
            delta = np.dot(self.weights[i].reshape(self.weights[i].shape[::-1]), delta)*(self.layer[i]*(1-self.layer[i]))
            i -= 1
        
        return delta
        
if __name__ == "__main__":
    # Creation of network with 4 layers, first one will have 784 neurons
    # second one will have 15 neurons and third one will also have 15 neurons
    # last one will have 10 neurons, it will represent digits from 0 to 9
    net = Network([784, 15, 15, 10])    
    with gzip.open("/Users/azizsattarov/Desktop/Federated Learning/neural-networks-and-deep-learning/data/mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    # val = cost_function_avg(train_set)

    # print(train_set[1])

    net.SGD(data_set=train_set, batch_size=100)

    # print(f"Total cost: {val}")

    
