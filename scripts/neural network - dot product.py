# Lesson1 Dot product for vectors
import numpy as np

inputs = [1, 2, 3, 6]
weights = [-0.1, 0.2, 0.6, 0.3]

bias = 0.3
outputs = np.dot(inputs, weights) + bias
# for each neuron, you can just have one bias, but you should have weights corresponding for each input.
# we have 4 inputs here, so for one neuron, we must have 4 weights for each input.

print(np.dot(inputs, weights) + bias)

# Try a layer with 3 neurons
# Three neurons with 3 rows of weights, one row stands for one set of weights for one neuron.

inputs = [1, 2, 3, 6]
weights = [[-0.1, 0.2, 0.6, 0.3],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.88]]
biases = [1, 2, 3]
# the shape of inputs (4*1 列向量)

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)
outputs = [np.dot(weights[0], inputs)+biases[0], np.dot(weights[1], inputs)+biases[1], np.dot(weights[2], inputs)+biases[2]]
print(outputs)


# Matrix Product
# To perform a matrix product, the size of the second dimension of the left matrix must match the size of the first
# dimension of the right matrix

# Transposition for the matrix product

# increase the input to batch of data
inputs = [[1,2,3,2.5], [2,5,-1,2], [-1.5, 2.7, 3.3,-1]]
weights = [[1,2,3,4], [0.5,1,1.5,2], [2,4,6,8]]
biases = [1.1, 2.2, 3.3]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
