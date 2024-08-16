from __future__ import annotations
from typing import NamedTuple, Sequence, Any

from scipy.sparse import csr_matrix, csr_array
import numpy as np

SparseMatrix = csr_matrix
SparseVector = csr_array

ExampleMultiLabel = Sequence[str]
BatchMultiLabel = Sequence[ExampleMultiLabel]


class Example(NamedTuple):
    """Single Cell Expression Example."""

    # A vector of size (num_genes, )
    gene_expression: SparseVector
    # A list of perturbations applied to the cell
    perturbations: Sequence[str]  ## TODO: Should be [] if control
    # A map from covariate name to covariate value
    covariates: dict[str, str] | None = None
    # A map from control condition name to control gene expression of
    # shape (num_controls_in_condition, num_genes)
    controls: SparseVector | None = None
    # A cell id
    id: str | None = None
    # A list of gene names of length num_genes
    gene_names: Sequence[str] | None = None
    # Optional foundation model embeddings
    embeddings: np.ndarray | None = None


class Batch(NamedTuple):
    """Single Cell Expression Batch."""

    gene_expression: SparseMatrix
    perturbations: Sequence[list[str]] | SparseMatrix
    covariates: dict[str, Sequence[str] | SparseMatrix] | None = None
    controls: SparseMatrix | None = None
    id: Sequence[str] | None = None
    gene_names: Sequence[str] | None = None
    embeddings: np.ndarray | None = None


class FrozenDictKeyMap(dict):
    """A dictionary that uses dictionaries as keys.

    Dictionaries cannot be used directly as keys to another dictionary because
    they are mutable. As a result this class first converts the dictionary to a
    frozenset of key-value pairs before using it as a key. The underlying data
    is stored using the dictionary data structure and this class just modifies
    the accessor and mutator methods.

    Example:
        >>> d = FrozenDictKeyMap()
        >>> d[{"a": 1, "b": 2}] = 1
        >>> d[{"a": 1, "b": 2}] = 2
        >>> d[{"a": 1, "b": 2}] = 3
        >>> d
        {frozenset({('a', 1), ('b', 2)}): 3}

    Attributes: see dict class
    """

    def __init__(self, data: Sequence[tuple[dict, Any]] | None = None):
        """Initialize the dictionary.

        Args:
            data: a sequence of (key, value) pairs to initialize the dictionary
        """
        if data is not None:
            try:
                _data = [(frozenset(key.items()), value) for key, value in data]
            except AttributeError as exc:
                raise ValueError(
                    "data must be a sequence of (key, value) pairs where key is a "
                    "dictionary"
                ) from exc
        else:
            _data = []
        super().__init__(_data)

    def __getitem__(self, key: dict) -> Any:
        """Get the value associated with the key.

        Args:
            key: a dictionary.

        Returns:
            The value associated with the key.
        """
        if isinstance(key, frozenset):
            key = dict(key)
        return super().__getitem__(frozenset(key.items()))

    def __setitem__(self, key: dict, value: Any) -> None:
        """Set the value associated with the key.

        Args:
            key: a dictionary.
            value: the value to set.
        """
        if isinstance(key, frozenset):
            key = dict(key)
        super().__setitem__(frozenset(key.items()), value)
