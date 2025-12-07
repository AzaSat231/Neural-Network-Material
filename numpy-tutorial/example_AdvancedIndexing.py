import numpy as np

palette = np.array([[0, 0, 0],         # 0 black
                    [255, 0, 0],       # 1 red
                    [0, 255, 0],       # 2 green
                    [0, 0, 255],       # 3 blue
                    [255, 255, 255]])  # 4 white
image = np.array([[0, 1, 2, 3, 4],  # each value corresponds to a color in the palette
                  [0, 3, 4, 0, 1]])

print(palette[image])  # the (2, 4, 3) color image




a = np.arange(12).reshape(3, 4)
print(a)

i = np.array([[0, 1],  # indices for the first dim of `a`
              [1, 2]])
j = np.array([[2, 1],  # indices for the second dim
              [3, 3]])

print(a[i, j])  # i and j must have equal shape, its like a map to a matrix 




a = np.arange(5)
print(a)

a[[1, 3, 4]] = 0

print(a)



a = np.arange(12).reshape(3, 4)
b = a > 4
print(b)


# We can use it, for example to say all values that higher 4 must be 0

a[b] = 0
print(a)