from math import *  # for debugging convenience
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from mininet.modules import LinearRegression, SoftmaxClassifier


def generate_training_data(num_examples: int, num_classes: int, num_features: int) -> Tuple[np.array, np.array]:
    """
    Builds a training dataset

    :param num_examples:
    :param num_classes:
    :param num_features:
    :return: (examples: num_examples X num_features, true labels: num_examples X num_classes)
    """
    assert num_examples % num_classes == 0

    # examples = np.random.random((num_examples, num_features))
    examples = np.ones((num_examples, num_features))
    class_ids = (
        np.arange(0, num_classes)
        .repeat(num_examples / num_classes)
    )
    examples = examples * class_ids.reshape(num_examples, 1)
    true_labels = np.zeros((num_examples, num_classes))
    true_labels[np.arange(class_ids.size), class_ids] = 1

    return examples, true_labels


def plot_regression_lines(weights) -> None:
    """
    Plots all regression lines represented by the weights.

    num_features should be equal to 3: x1, x2, b
    num_outputs denotes the number of regression lines to draw.

    :param weights: (num_features, num_outputs)
    """
    assert weights.shape[0] == 3

    w11 = weights[0][0]
    w12 = weights[1][0]
    b1 = weights[2][0]
    w21 = weights[0][1]
    w22 = weights[1][1]
    b2 = weights[2][1]

    x1 = np.arange(0, 5, 0.01)
    x2 = np.arange(0, 5, 0.01)
    pred1 = w11 * x1 + w12 * x2 + b1
    pred2 = w21 * x1 + w22 * x2 + b2
    balance = ((w11 - w21) * x1 + b1 - b2) / (w22 - w12)
    # plt.fill_between(x1, 0, balance, color='#ffcccc')  # red
    # plt.fill_between(x1, balance, 5, color='#99ccff')  # blue

    plt.plot(x1, balance)

    # for regression in weights.T:
    #     w1, w2, b = regression
    #     p1 = (0, b / (-w2))
    #     p2 = (1, (w1 + b) / (-w2))
    #     plt.axline(p1, p2)


g_fig = None


def main():
    np.random.seed(0)

    num_examples = 4
    num_classes = 2
    num_features = 2

    epochs = 100
    learning_rate = 1

    # examples, labels = generate_training_data(num_examples, num_classes, num_features)
    examples = np.array([
        [2, 1],
        [3, 2],
        [4, 3],

        [1, 2],
        [2, 3],
        [3, 4],
    ])
    labels = np.array([
        [1, 0],
        [1, 0],
        [1, 0],

        [0, 1],
        [0, 1],
        [0, 1],
    ])

    linear_regression_layer = LinearRegression(
        num_features=num_features,
        num_outputs=num_classes,
        weights_initializer=lambda x, y: np.ones((x, y)),
        biases_initializer=lambda x, y: np.ones((x, y))
    )
    classification_layer = SoftmaxClassifier(num_classes=num_classes)
    # plt.ion()

    def draw_state():
        # global g_fig
        # if g_fig is not None:
        #     plt.close(g_fig)
        plt.close('all')
        g_fig = plt.figure()
        for point, label in zip(examples, labels):
            color = 'r' if label[0] else 'b'
            plt.plot(point[0], point[1], color + 'd')

        plot_regression_lines(linear_regression_layer.weights)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        plt.show()

    last_loss = None
    for epoch in range(epochs):
        print('--- Epoch: ', epoch)

        # forward pass
        l1_pred = linear_regression_layer.forward(examples)
        l2_pred = classification_layer.forward(l1_pred)
        loss_per_example = classification_layer.loss(labels)
        assert (loss_per_example >= 0).all()
        loss = np.mean(loss_per_example)
        if last_loss is not None:
            assert loss < last_loss
        last_loss = loss
        print('Loss: ', loss)

        # backward pass
        l2_grad = classification_layer.grad_weights(labels)
        l1_grad = linear_regression_layer.grad_weights(l2_grad)

        # # average gradient over all examples
        # update = np.mean(l1_grad, axis=0)
        # # alternative update
        # alt_update = np.matmul(np.hstack((examples, np.ones((examples.shape[0], 1)))).T, l2_grad) / len(examples)

        linear_regression_layer.update(-1 * learning_rate * l1_grad)
        draw_state()

    print('--- Test after training')
    # forward pass
    result = classification_layer.forward(linear_regression_layer.forward(examples))
    print('Loss: ', np.mean(classification_layer.loss(labels)), '; Result: ', result)


if __name__ == '__main__':
    main()
