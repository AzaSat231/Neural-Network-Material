import gzip
import pickle

with gzip.open("/Users/azizsattarov/Desktop/Federated Learning/neural-networks-and-deep-learning/data/mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

# train_set = (X_train, y_train) X_train is array of images, y_train is array of labels
# So using first [0] is represent that we're playing with X_train
print(len(test_set[0][0]))  
