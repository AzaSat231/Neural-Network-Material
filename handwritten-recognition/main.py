import numpy as np

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

# Representation of neural network, it initialization
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #Bias and weight are initialised randomly, but there is a better way to do it
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
        


if __name__ == "__main__":
    # Creation of network with 3 layers, first one will have 2 neurons
    # second one will have 3 neurons and last one will have 1 neuron
    net = Network([2, 3 ,1])    
    print(net.weights)
    print(net.weights[1])   #Weights for 2 and 3 layer of neurons