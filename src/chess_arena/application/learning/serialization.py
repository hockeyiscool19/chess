"""Convert networks to and from in-memory ``.npz`` bytes (never pickles)."""

import io
from typing import TYPE_CHECKING

import numpy as np

from chess_arena.application.learning.network import Array, ValueNetwork
from chess_arena.application.ports.training import NetworkBlob

if TYPE_CHECKING:
    from collections.abc import Callable


def to_npz_bytes(network: ValueNetwork) -> bytes:
    """Return the network's arrays as ``.npz`` bytes."""
    buffer = io.BytesIO()
    savez: Callable[..., None] = np.savez
    savez(buffer, **network.arrays())
    return buffer.getvalue()


def from_npz_bytes(data: bytes) -> dict[str, Array]:
    """Return named float32 arrays from ``.npz`` bytes, refusing pickled objects."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}


def to_blob(network: ValueNetwork) -> NetworkBlob:
    """Return a picklable blob of ``network``."""
    return NetworkBlob(npz=to_npz_bytes(network), activation=network.activation)


def from_blob(blob: NetworkBlob) -> ValueNetwork:
    """Rebuild a network from a blob."""
    return ValueNetwork.from_arrays(from_npz_bytes(blob.npz), blob.activation)
