from abc import ABC, abstractmethod
from typing import Sequence, TypeVar

from ..types import Example, Batch

Datum = TypeVar("Datum", Example, Batch)


class Transform(ABC):
    """Abstract transform interface."""

    @abstractmethod
    def __call__(self, data: Datum) -> Datum:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        return self.__class__.__name__ + "({!s})"


class ExampleTransform(Transform):
    """Transforms an example."""

    @abstractmethod
    def __call__(self, example: Datum) -> Datum:
        pass

    def __repr__(self) -> str:
        _base = super().__repr__()
        return "[example-wise]" + _base

    @classmethod
    def batchify(cls):
        """Converts an example transform to a batch transform."""
        raise NotImplementedError


class Dispatch(Transform):
    """Dispatches a transform to an example based on a key field.

    Attributes:
        transforms: A map of key to transform.
    """

    transforms: dict[str, Transform]

    def __init__(self, **transforms: Transform):
        """Initializes the instance based on passed transforms.

        Args:
            **transforms: A map of key to transform.
        """
        self.transforms = transforms

    def __call__(self, data: Datum) -> Datum:
        """Apply each transform to the field of an example matching its key."""
        result = {}
        try:
            for key, transform in self.transforms.items():
                result[key] = transform(getattr(data, key))
        except KeyError as exc:
            raise TypeError(
                f"Invalid {key=} in transforms. All keys need to match the "
                f"fields of an example."
            ) from exc

        return data._replace(**result)

    def __repr__(self) -> str:
        _base = super().__repr__()
        transforms_repr = ", ".join(
            f"{key}: {repr(transform)}" for key, transform in self.transforms.items()
        )
        return _base.format(transforms_repr)


class Compose(Transform):
    """Creates a transform from a sequence of transforms."""

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, data: Datum) -> Datum:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        transforms_repr = " \u2192 ".join(
            repr(transform) for transform in self.transforms
        )
        return "[{}]".format(transforms_repr)
