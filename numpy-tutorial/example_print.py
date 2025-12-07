import numpy as np

print(np.arange(10000).reshape(100, 100))

a = np.arange(15).reshape(3, 5)
print(a.dtype)

#In list it would just join lists but here it actually plus values inside
b = np.array([6, 7, 8])
print(b + np.array([1, 2, 3]))

c = np.array([1.2, 3.5, 5.1])
print(c.dtype)

# a = np.array(1, 2, 3, 4)    # WRONG
# Traceback (most recent call last):

#The type of the array can also be explicitly specified at creation time:
c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)

print(np.zeros((3, 4)))

print(f"Create sequences of numbers: {np.arange(10, 30, 5)}")

print(np.linspace(0, 2, 9))                   # 9 numbers from 0 to 2
