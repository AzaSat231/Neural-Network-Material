import numpy as np

rg = np.random.default_rng()   # create random generator
a = np.floor(10 * rg.random((3, 4)))
print(a)

print(a.ravel()) #return flattend array

# print(a.resize((2, 6)))



a = np.floor(10 * rg.random((2, 12)))

print(a)
# Split `a` into 3
print(np.hsplit(a, 6))





# Copies and views
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a            # no new object is created
print(b is a)           # a and b are two names for the same ndarray object



c = a.view()
print(c is a)
print(c.base is a )     # c is a view of the data owned by a