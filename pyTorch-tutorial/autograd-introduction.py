# torch.autograd is PyTorch’s automatic differentiation engine that powers neural network training.
import torch
from torchvision.models import resnet18, ResNet18_Weights

# Pre-trained model from torchvision
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Stored in 4D dimensions (batch_size, channels - one layer of information in an image (or feature map), height, width)
# So channel is a number of features of photo and
# Each feature in our example is 64×64 grid
# These are NOT colors — they are learned patterns (edges, shapes, etc.)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Run data through model, to make prediction
prediction = model(data)

loss = (prediction - labels).sum()

# Backpropagate this error through the network
loss.backward() # backward pass

# Optimizer updates the model’s parameters (weights and biases) to make the model learn.
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Updates all the model’s parameters (weights & biases) using the gradients that were computed during loss.backward().
optim.step() #gradient descent
