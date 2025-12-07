# Tensors are a specialized data structure that are very similar to arrays and matrices. 
# In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)




x_ones = torch.ones_like(x_data) # retains the properties of x_data (x_data will keep same shape as x_ones)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")




shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")




# tensor gonna be stored in cpu, but can be stored on gpu 
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")




# Use same slicing methods as numpy
tensor = torch.ones(4, 4)
tensor[:,2] = 0
print(tensor)




# u can use cat function to concatenate(connect together a sequence of tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)



# Operations that have a _ suffix are in-place, it will change x based on function
print(tensor, "\n")
tensor.add_(5)
print(tensor)





# Tensors on the CPU and NumPy arrays can share their underlying memory locations, 
# and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")



# Change in tensor will reflect numpy 
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")



# Also it works vie versa tensor can be created from numpy 
# and change in numpy will change tensor 
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")