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





if __name__ == "__main__":
    transform = transforms.ToTensor()
    
    # Load dataset, into two sets: trainset, testset
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
    train_dataloader = DataLoader(mnist_trainset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

    # Create our training model
    model = NeuralNetwork()
    # print(model)













    # ----------------------------------------------------------
    # Explnation of how Neural Network class work and intialised
    # ----------------------------------------------------------

    input_image = torch.rand(3,28,28)
    # print(input_image.size())

    # We use flatten to convert our 2D 28x28 into image into a contiguous array of 784 pixel values 
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    # print(flat_image.size())

    # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    # Here it initialises all biases and weights. 
    # It doing this: input nodes  ──(weights W)──▶ hidden nodes
    hidden1 = layer1(flat_image)
    print(hidden1.size())

    """ Non-linear activations are what create the complex mappings between the model’s inputs and outputs. 
        They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
        In easier words it just apply this ReLU(x)=max(0,x). 
        So now it will do this: Input → [ Linear (W, b) ] → [ ReLU ] → Output """
    # print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")

    # nn.Sequential is an ordered container of modules. 
    # The data is passed through all the modules in the same order as defined. 
    # You can use sequential containers to put together a quick network like seq_modules.
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)





    # ----------------------------------------------------------------------
    # Explnation of how backpropogation work in pyTorch using torch.autograd
    # ----------------------------------------------------------------------

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    """ This object knows how to compute the function in the forward direction, 
        and also how to compute its derivative during the backward propagation step. 
        A reference to the backward propagation function is stored in grad_fn property of a tensor. """
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    """ By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. 
        However, there are some cases when we do not need to do that, for example, 
        when we have trained the model and just want to apply it to some input data, 
        i.e. we only want to do forward computations through the network.
        So we can disable it by torch.no_grad() """
    z = torch.matmul(x, w)+b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print(z.requires_grad)