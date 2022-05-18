import numpy as np


class SoftmaxClassifier:

    def __init__(
            self,
            *,
            num_classes: int
    ):
        self.num_classes = num_classes
        self._softmax = None

    def softmax(self, inputs: np.array) -> np.array:
        """
        Compute the forward pass.

        :param inputs: num_examples X num_classes
        :return: num_examples X num_classes
        """
        assert self.num_classes == inputs.shape[1]

        exp = np.exp(inputs)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, inputs: np.array) -> np.array:
        """
        Compute the forward pass.

        :param inputs: num_examples X num_classes
        :return: class scores: num_examples X num_classes
        """
        self._softmax = self.softmax(inputs)
        return self._softmax

    def loss(self, true_labels: np.array) -> np.array:
        """
        Compute the Categorical Cross-Entropy Loss

        :param true_labels: num_examples X num_classes
        :return: loss: num_examples X 1
        """
        assert self._softmax is not None, 'forward() must be called first'
        assert self._softmax.shape == true_labels.shape

        return -1 * np.sum(true_labels * np.log(self._softmax), axis=1, keepdims=True)

    def grad_weights(self, true_labels: np.array) -> np.array:
        """
        Compute gradients for trainable parameters during the backward pass.
        :param true_labels: num_examples X num_classes
        :return: num_examples X num_classes
        """
        assert self._softmax is not None, 'forward() must be called first'
        assert self._softmax.shape == true_labels.shape

        return self._softmax - true_labels
