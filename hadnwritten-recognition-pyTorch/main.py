import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # nn.Sequential is an ordered container of modules. 
        # The data is passed through all the modules in the same order as defined.
        # In this nn.Linear we gonna have weights, biases and gradient. It will look like:
        """ model.weight.data   # actual values
            model.weight.grad   # gradient (after backward) 
            model.layer1.weight.grad  # now contains ∂loss/∂W1
            model.layer1.bias.grad """
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # Forward pass — no gradients yet
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation, so it applies the chain rule backward for every parameter in the graph.
        # So it will not store gradients into loss, it will store it model parameters in nn.Linear in our class, parameter.grad
        # model.weight.data - actual values
        # model.weight.grad - gradient (after backward)
        loss.backward()

        # Actual learning happening here, so it calculate our new new weights and biases
        # based on the loss parameters that was calculated
        optimizer.step()

        # To not sum up gradients from before echo we need to initialise it
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == "__main__":
    transform = transforms.ToTensor()
    
    # Load dataset, into two sets: trainset, testset
    # It will not download it again, if it already exist
    mnist_trainset = datasets.MNIST(
        root="/Users/azizsattarov/Desktop/Federated Learning",
        train=True,
        download=True,
        transform=transform
    )

    mnist_testset = datasets.MNIST(
        root="/Users/azizsattarov/Desktop/Federated Learning",
        train=False,
        download=True,
        transform=transform
    )

    # Initialise what the size of minibatches will be 64
    train_dataloader = DataLoader(mnist_trainset, batch_size=64)
    test_dataloader = DataLoader(mnist_testset, batch_size=64)

    # Create our training model
    model = NeuralNetwork()

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    # Initialize the loss function (same as cost function, both calculate same thing)
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer store references to our weights and biases. 
    # Not copies - actual references to nn.Parameter objects inside your model.
    # So you can change weights and biases inside optimizer and it automatically will be changed in our model
    # optimizer → [W1, b1, W2, b2, ...]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")




