import numpy as np


class Module:

    def forward(self, inputs: np.array) -> np.array:
        pass

    def grad_weights(self, errors: np.array) -> np.array:
        pass

    # def update(self, gradient: np.array) -> None:
    #     pass
