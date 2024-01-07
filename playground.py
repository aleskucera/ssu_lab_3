import numpy as np

arr = np.array([[4, 2, 3],
                [1, 5, 6]])

print(arr.shape)
print(arr)

# Get the argmax of each row
print(np.argmax(arr, axis=0))
