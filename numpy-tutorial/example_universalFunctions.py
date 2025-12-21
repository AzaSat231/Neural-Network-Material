import numpy as np

B = np.arange(3)

print(np.exp(B))

print(np.sqrt(B))

arr1 = [[1, 2], 
        [3, 4]]
arr2 = [[5, 6], 
        [7, 8]]

print(np.dot(arr1, arr2))

array_2d = np.array([[1, 2], [3, 4]])
transposed_array = np.transpose(array_2d)
print(transposed_array)