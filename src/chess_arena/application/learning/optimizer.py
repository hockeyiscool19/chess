"""Adam with global gradient-norm clipping and decoupled weight decay."""

import math

import numpy as np

from chess_arena.application.learning.network import Array


class Adam:
    """Adam over a fixed list of parameter arrays, updated in place."""

    def __init__(
        self,
        parameters: list[Array],
        learning_rate: float,
        weight_decay: float = 0.0,
        max_norm: float = 1.0,
    ) -> None:
        """Allocate first and second moments for every parameter."""
        self._parameters = parameters
        self.learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._max_norm = max_norm
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._epsilon = 1e-8
        self._first = [np.zeros_like(parameter) for parameter in parameters]
        self._second = [np.zeros_like(parameter) for parameter in parameters]
        self._steps = 0

    def step(self, grads: list[Array]) -> float:
        """Apply one update and return the pre-clipping gradient norm."""
        norm = math.sqrt(sum(float((grad * grad).sum()) for grad in grads))
        scale = min(1.0, self._max_norm / (norm + 1e-12))
        self._steps += 1
        correction1 = 1.0 - self._beta1**self._steps
        correction2 = 1.0 - self._beta2**self._steps
        rate = self.learning_rate * math.sqrt(correction2) / correction1
        for parameter, grad, first, second in zip(
            self._parameters, grads, self._first, self._second, strict=True
        ):
            clipped = grad * scale
            _ = np.multiply(first, self._beta1, out=first)
            _ = np.add(first, (1.0 - self._beta1) * clipped, out=first)
            _ = np.multiply(second, self._beta2, out=second)
            _ = np.add(second, (1.0 - self._beta2) * clipped * clipped, out=second)
            if self._weight_decay and parameter.ndim > 1:
                decay = self.learning_rate * self._weight_decay * parameter
                _ = np.subtract(parameter, decay, out=parameter)
            step = rate * first / (np.sqrt(second) + self._epsilon)
            _ = np.subtract(parameter, step, out=parameter)
        return norm
