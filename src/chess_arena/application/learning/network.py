"""A small multilayer perceptron in numpy: 768 inputs, hidden layers, tanh output.

The output is the expected game result for the side to move, from -1 (loss)
through 0 (draw) to +1 (win). Hidden layers use a clipped ReLU (values held in
[0, 1], as in NNUE evaluators) or tanh. Weights are plain float32 arrays so a
model file is just an ``.npz`` archive.
"""

import math
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from chess_arena.application.learning.features import FEATURES
from chess_arena.domain.recipes import Activation, NetworkSpec

Array = NDArray[np.float32]
_ACTIVE_PIECES = 32.0
_OUTPUT_STD = 0.01


def _activate(z: Array, activation: Activation) -> Array:
    """Apply the hidden nonlinearity."""
    if activation is Activation.CRELU:
        return np.clip(z, 0.0, 1.0)
    return np.tanh(z)


def _derivative(z: Array, a: Array, activation: Activation) -> Array:
    """Return d(activation)/dz given the pre-activation and activation."""
    if activation is Activation.CRELU:
        return ((z > 0.0) & (z < 1.0)).astype(np.float32)
    return (1.0 - a * a).astype(np.float32)


class ForwardCache:
    """Inputs, pre-activations, and activations kept for backpropagation."""

    def __init__(self, inputs: Array, pre: list[Array], post: list[Array]) -> None:
        """Store the per-layer tensors of one forward pass."""
        self.inputs = inputs
        self.pre = pre
        self.post = post


class ValueNetwork:
    """Weights and biases for every layer plus the hidden activation."""

    def __init__(self, weights: list[Array], biases: list[Array], activation: Activation) -> None:
        """Bind layer parameters; the last layer must have one output."""
        if len(weights) != len(biases) or weights[-1].shape[1] != 1:
            msg = "a value network needs matching layers and a single output"
            raise ValueError(msg)
        if weights[0].shape[0] != FEATURES:
            msg = f"the first layer must take {FEATURES} inputs"
            raise ValueError(msg)
        self.weights = weights
        self.biases = biases
        self.activation = activation

    @classmethod
    def initialize(cls, spec: NetworkSpec, rng: np.random.Generator) -> "ValueNetwork":
        """Return random hidden layers and a near-zero output layer."""
        sizes = [FEATURES, *spec.hidden, 1]
        weights: list[Array] = []
        biases: list[Array] = []
        for index, (fan_in, fan_out) in enumerate(pairwise(sizes)):
            last = index == len(sizes) - 2
            effective = _ACTIVE_PIECES if index == 0 else float(fan_in)
            std = _OUTPUT_STD if last else 1.0 / math.sqrt(effective)
            weights.append(rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32))
            biases.append(np.zeros(fan_out, dtype=np.float32))
        return cls(weights, biases, spec.activation)

    @property
    def spec(self) -> NetworkSpec:
        """Return the architecture as a network spec."""
        hidden = tuple(int(weight.shape[1]) for weight in self.weights[:-1])
        return NetworkSpec(hidden=hidden, activation=self.activation)

    def parameters(self) -> list[Array]:
        """Return parameters in optimizer order (weight, bias per layer)."""
        return [array for pair in zip(self.weights, self.biases, strict=True) for array in pair]

    def value(self, indices: list[int]) -> float:
        """Return the value of one position from its active feature indices."""
        hidden = self.weights[0][indices].sum(axis=0) + self.biases[0]
        activation = _activate(hidden, self.activation)
        for weight, bias in zip(self.weights[1:-1], self.biases[1:-1], strict=True):
            activation = _activate(activation @ weight + bias, self.activation)
        output = activation @ self.weights[-1] + self.biases[-1]
        return math.tanh(float(output[0]))

    def forward(self, inputs: Array) -> tuple[Array, ForwardCache]:
        """Return batch outputs in (-1, 1) and the cache for backpropagation."""
        pre: list[Array] = []
        post: list[Array] = []
        activation = inputs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = (activation @ weight + bias).astype(np.float32)
            last = index == len(self.weights) - 1
            activation = np.tanh(z) if last else _activate(z, self.activation)
            pre.append(z)
            post.append(activation)
        return post[-1][:, 0], ForwardCache(inputs, pre, post)

    def backward(self, cache: ForwardCache, grad_output: Array) -> list[Array]:
        """Return parameter gradients given dLoss/dOutput for each batch row."""
        grads: list[Array] = []
        delta = (grad_output[:, None] * (1.0 - cache.post[-1] ** 2)).astype(np.float32)
        for index in range(len(self.weights) - 1, -1, -1):
            below = cache.inputs if index == 0 else cache.post[index - 1]
            grads.append(delta.sum(axis=0))
            grads.append((below.T @ delta).astype(np.float32))
            if index > 0:
                upstream = delta @ self.weights[index].T
                delta = (
                    upstream
                    * _derivative(cache.pre[index - 1], cache.post[index - 1], self.activation)
                ).astype(np.float32)
        grads.reverse()
        return grads

    def arrays(self) -> dict[str, Array]:
        """Return named arrays for an ``.npz`` archive."""
        named: dict[str, Array] = {}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            named[f"w{index}"] = weight
            named[f"b{index}"] = bias
        return named

    @classmethod
    def from_arrays(cls, arrays: dict[str, Array], activation: Activation) -> "ValueNetwork":
        """Rebuild a network from :meth:`arrays` output."""
        count = sum(1 for name in arrays if name.startswith("w"))
        weights = [arrays[f"w{index}"].astype(np.float32) for index in range(count)]
        biases = [arrays[f"b{index}"].astype(np.float32) for index in range(count)]
        return cls(weights, biases, activation)

    def copy(self) -> "ValueNetwork":
        """Return an independent deep copy."""
        return ValueNetwork(
            [weight.copy() for weight in self.weights],
            [bias.copy() for bias in self.biases],
            self.activation,
        )
