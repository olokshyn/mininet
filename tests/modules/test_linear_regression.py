import numpy as np

from mininet.modules.linear_regression import LinearRegression


def initializer_zeros(n_rows, n_cols):
    return np.zeros((n_rows, n_cols))


def initializer_ones(n_rows, n_cols):
    return np.ones((n_rows, n_cols))


def initializer_range(n_rows, n_cols):
    return np.arange(1, n_rows * n_cols + 1).reshape(n_rows, n_cols)


def test_forward_zeros():
    lr = LinearRegression(
        num_features=4,
        num_outputs=2,
        weights_initializer=initializer_zeros,
        biases_initializer=initializer_zeros,
    )

    assert (np.zeros((3, 2)) == lr.forward(np.ones((3, 4)))).all()
    assert (np.zeros((15, 2)) == lr.forward(np.arange(10, 70).reshape(15, 4))).all()

    lr = LinearRegression(
        num_features=4,
        num_outputs=2,
        weights_initializer=initializer_zeros,
        biases_initializer=initializer_ones,
    )

    assert (np.ones((3, 2)) == lr.forward(np.ones((3, 4)))).all()
    assert (np.ones((15, 2)) == lr.forward(np.arange(10, 70).reshape(15, 4))).all()


def test_forward_simple():
    lr = LinearRegression(
        num_features=3,
        num_outputs=2,
        weights_initializer=initializer_range,
        biases_initializer=initializer_range,
    )

    inputs = np.array([
        [3, 4, 5],
        [1, 2, 3],
        [2, 3, 4],
    ])
    outputs = np.array([
        [
            np.dot([3, 4, 5], [1, 3, 5]) + 1,
            np.dot([3, 4, 5], [2, 4, 6]) + 2,
        ],
        [
            np.dot([1, 2, 3], [1, 3, 5]) + 1,
            np.dot([1, 2, 3], [2, 4, 6]) + 2,
        ],
        [
            np.dot([2, 3, 4], [1, 3, 5]) + 1,
            np.dot([2, 3, 4], [2, 4, 6]) + 2,
        ],
    ])
    assert (outputs == lr.forward(inputs)).all()


def test_grad_weights_simple():
    lr = LinearRegression(
        num_features=4,
        num_outputs=3,
        weights_initializer=initializer_range,
        biases_initializer=initializer_range
    )

    inputs = np.arange(1, 9).reshape(2, 4)
    outputs = np.array([
        [
            np.dot([1, 2, 3, 4], [1, 4, 7, 10]) + 1,
            np.dot([1, 2, 3, 4], [2, 5, 8, 11]) + 2,
            np.dot([1, 2, 3, 4], [3, 6, 9, 12]) + 3,
        ],
        [
            np.dot([5, 6, 7, 8], [1, 4, 7, 10]) + 1,
            np.dot([5, 6, 7, 8], [2, 5, 8, 11]) + 2,
            np.dot([5, 6, 7, 8], [3, 6, 9, 12]) + 3,
        ]
    ])

    assert (outputs == lr.forward(inputs)).all()

    errors = np.arange(1, 7).reshape(2, 3)
    # input: num_examples X num_features
    # weights: num_features X num_outputs
    # output: num_examples X num_outputs
    # errors: num_examples X num_outputs
    # gradient: num_examples X num_features X num_outputs
    gradient = np.array([
        np.array([
            np.array([1, 2, 3, 4, 1]) * 1,
            np.array([1, 2, 3, 4, 1]) * 2,
            np.array([1, 2, 3, 4, 1]) * 3,
        ]).T,
        np.array([
            np.array([5, 6, 7, 8, 1]) * 4,
            np.array([5, 6, 7, 8, 1]) * 5,
            np.array([5, 6, 7, 8, 1]) * 6,
        ]).T,
    ])

    assert gradient.shape[1] == lr.weights.shape[0]
    assert gradient.shape[2] == lr.weights.shape[1]
    assert (gradient == lr.grad_weights(errors)).all()
