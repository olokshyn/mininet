from typing import Callable

import numpy as np

from .module import Module


class LinearRegression(Module):

    def __init__(
            self,
            *,
            num_features: int,
            num_outputs: int,
            weights_initializer: Callable[[int, int], np.array] = np.random.rand,
            biases_initializer: Callable[[int, int], np.array] = np.random.rand
    ):
        self.num_features = num_features
        self.num_outputs = num_outputs
        weights = weights_initializer(num_features, num_outputs)
        biases = biases_initializer(1, num_outputs)
        self.weights = np.vstack((weights, biases))
        self._saved_inputs = None

    def forward(self, inputs: np.array) -> np.array:
        """
        Compute the forward pass.

        :param inputs: num_examples X num_features
        :return: num_examples X num_outputs
        """
        assert inputs.shape[1] == self.num_features
        inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))

        self._saved_inputs = inputs
        return np.matmul(inputs, self.weights)

    def grad_weights(self, errors: np.array) -> np.array:
        """
        Compute gradients for trainable parameters during the backward pass.

        :param errors: num_examples X num_outputs
        :return: num_examples X num_features X num_outputs
        """

        assert self._saved_inputs is not None, 'forward() must be called first'
        assert errors.shape[0] == self._saved_inputs.shape[0]
        assert errors.shape[1] == self.num_outputs

        # return self._saved_inputs[..., None] * errors[:, None, :]
        return np.matmul(self._saved_inputs.T, errors) / len(self._saved_inputs)

    def update(self, update: np.array) -> None:
        assert self.weights.shape == update.shape
        self.weights += update
