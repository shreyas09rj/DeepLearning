import numpy as np

image = np.array([
    [1,3,2,4],
    [5,6,7,8],
    [9,2,1,3],
    [4,5,6,7]
])

pool_size = 2
stride = 2

output_size = (image.shape[0] - pool_size) // stride + 1

output = np.zeros((output_size,output_size))

for i in range(output_size):
    for j in range(output_size):
        region = image[
            i*stride:i*stride+pool_size,
            j*stride:j*stride+pool_size
        ]
        output[i,j] = np.max(region)
print("input:")
print(image)

print("Max pool:")
print(output)