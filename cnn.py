import numpy as np
from layer import Layer
from scipy import signal


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.input_depth = input_shape
        self.input_height = input_shape
        self.depth = depth

        self.kernels_shape = (depth, input_shape, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)

        self.output_shape = (depth, input_shape - kernel_size + 1, input_shape - kernel_size + 1)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self):
        # TODO: update definition
        pass

    def backward(self):
        # TODO: update definition
        pass
