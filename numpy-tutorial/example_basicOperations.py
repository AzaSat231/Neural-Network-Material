import numpy as np

a = np.array([20, 30, 40, 50]).reshape(4, 1)
b = np.arange(4).reshape(4,1)

print(a)

print(a - b)

print(a*b)

print(a < 35)

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
print(A * B)

rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3

print(a)

print(b + a)



a = rg.random((2, 3))

print(a)

print(a.sum())

print(a.min())




b = np.arange(12).reshape(3, 4)

print(f"b value: {b}")

print(b.sum(axis=0))

print(b.sum(axis=1))