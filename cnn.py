import numpy as np
from base_layer import BaseLayer
from scipy import signal


class ConvolutionalLayer(BaseLayer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.input_depth = input_shape
        self.input_height = input_shape
        self.depth = depth

        self.kernels_shape = (depth, input_shape, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)

        self.output_shape = (depth, input_shape - kernel_size + 1, input_shape - kernel_size + 1)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'valid')
                return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.input_depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], kernels_gradient[i, j], mode='full')

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
