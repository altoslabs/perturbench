from typing import Any, Callable

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .base import Transform


class ToDense(Transform):
    """Convert a sparse matrix/tensor to a dense matrix/tensor."""

    def __call__(self, value: torch.Tensor | csr_matrix | np.ndarray) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            output = value.to_dense()
        elif isinstance(value, csr_matrix):
            output = torch.Tensor(value.toarray())
        elif isinstance(value, np.ndarray):
            output = torch.Tensor(value)
        else:
            raise TypeError(
                f"Invalid type for {value}. Must be either a tensor or a csr_matrix."
            )
        return output

    def __repr__(self):
        return "ToDense"


class ToFloat(Transform):
    """Convert a tensor to float."""

    def __call__(self, value: torch.Tensor | np.ndarray | list):
        if isinstance(value, torch.Tensor):
            return value.float()
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype(np.float32))
        elif isinstance(value, list):
            return torch.from_numpy(np.array(value).astype(np.float32))
        else:
            raise TypeError(
                f"Invalid type for {value}. Must be either a tensor or a numpy array."
            )

    def __repr__(self):
        return "ToFloat"


class Unsqueeze(Transform):
    """Add a dimension at the specified position."""

    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(self.dim)

    def __repr__(self):
        return f"Unsqueeze(dim={self.dim})"


class MapApply(Transform):
    """Map each transform to an input based on a key.

    Attributes:
        transform_map: A map of key to transform.
    """

    transform_map: dict[str, Transform | Callable]

    def __init__(
            self,
            transforms: dict[str, Transform | Callable],
            init_params_map: dict | None = None,
    ) -> None:
        """Initializes the instance based on passed transforms.

        This classes supports two ways of initializing the transforms. The first
        is by passing a map of key to transform. The second is by passing a map of
        key to factory callable. The factory callable will be called with the
        corresponding init params from the init_params_map. The factory callable
        should return a Transform.

        Args:
            transforms: A map of key to transform.
            init_params_map: A map of key to init params for the transforms.

        Raises:
            ValueError: If init_params_map is not None when using a dict of
                Transforms.
            TypeError: If the transform is not a dict of Transform or a callable.
        """
        super().__init__()
        self.transform_map = {}
        for key, transform in transforms.items():
            # Transforms are dict[str, Transform], directly assign them
            if isinstance(transform, Transform):
                if init_params_map is not None:
                    raise ValueError(
                        "init_params_map should be None when using a dict of "
                        "Transforms."
                    )
                self.transform_map[key] = transform
            # Transforms are dict[str, factory_callable], call the factory
            elif callable(transform):
                self.transform_map[key] = transform(init_params_map[key])
            else:
                raise TypeError(
                    f"Invalid type for {key=} in transform. Must be either a "
                    f"Transform or a callable."
                )

    def __call__(self, value_map: dict[str, Any]) -> dict[str, Any]:
        return {key: self.transform_map[key](val) for key, val in value_map.items()}

    def __repr__(self) -> str:
        transforms_repr = ", ".join(
            f"{key}: {repr(transform)}" for key, transform in self.transform_map.items()
        )
        return "{" + transforms_repr + "}"
