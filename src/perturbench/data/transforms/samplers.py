import numpy as np
import torch
from scipy.sparse import csr_matrix
from typing import Sequence

from .base import ExampleTransform

class RowSampler(ExampleTransform):
    """
    Sample a fixed number of rows from a matrix.

    Attributes:
        num_samples: number of rows to sample.
        seed: the seed for the random number generator.
    """

    def __init__(self, num_samples: int = 1, seed: int | None = None):
        self.num_samples = num_samples
        self._rng = np.random.default_rng(seed)

    def __call__(self, controls: Sequence[int] | csr_matrix) -> torch.Tensor:
        if isinstance(controls, csr_matrix):
            num_controls = controls.shape[0]
        else:
            num_controls = len(controls)
            
        sampled_indices = self._rng.integers(num_controls, size=self.num_samples)
        return controls[sampled_indices]

    def __repr__(self):
        _base = super().__repr__()
        return _base.format(f"num_samples={self.num_samples}")
