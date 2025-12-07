import numpy as np

a = np.arange(10)**3

print(a)

print(a[2:5])

print(a[::-1])

print(a[-1])


def f(x, y):
    return 10 * x + y

b = np.fromfunction(f, (4, 5), dtype=int)

print(b)

for row in b.flat:
    print(row)