import numpy as np
import random
import sys
import gzip
import pickle



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
        print(f"Bias: {self.biases}")
        print(f"Weight: {self.weights}")
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            print(f"\nLayer {i+1}:")
            print(f"Bias in loop: {b}")
            print(f"Weight in loop: {w}")
            a = sigmoid(np.dot(w, a)+b)
            print(f"a: {a}")
        
        return a


if __name__ == "__main__":
    # Creation of network with 3 layers, first one will have 2 neurons
    # second one will have 3 neurons and last one will have 1 neuron
    net = Network([784, 15, 15, 10])    
    with gzip.open("/Users/azizsattarov/Desktop/Federated Learning/neural-networks-and-deep-learning/data/mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    imageValue = np.array(train_set[0][0])

    imageValue = imageValue.reshape(784, 1)    

    print(f"Value: {imageValue.shape}")

    net.feedforward(imageValue)

    

    
