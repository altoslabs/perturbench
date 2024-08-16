import itertools
import functools
from typing import Collection, Sequence

import torch
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer

from .base import Transform
from ..types import ExampleMultiLabel, BatchMultiLabel


class OneHotEncode(Transform):
    """One-hot encode a categorical variable.

    Attributes:
        onehot_encoder: the wrapped encoder instance
    """

    one_hot_encoder: OneHotEncoder

    def __init__(self, categories: Collection[str], **kwargs):
        categories = [list(categories)]
        self.one_hot_encoder = OneHotEncoder(
            categories=categories,
            sparse_output=False,
            **kwargs,
        )

    def __call__(self, labels: Sequence[str]):
        string_array = np.array(labels).reshape(-1, 1)
        encoded = self.one_hot_encoder.fit_transform(string_array)
        return torch.Tensor(encoded)

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.one_hot_encoder.categories[0])
        return _base.format(categories)


class LabelEncode(Transform):
    """Label encode categorical variables.

    Attributes:
        ordinal_encoder: sklearn.preprocessing.OrdinalEncoder
    """

    ordinal_encoder: OrdinalEncoder

    def __init__(self, values: Sequence[str]):
        categories = [np.array(values)]
        self.ordinal_encoder = OrdinalEncoder(categories=categories)

    def __call__(self, labels: Sequence[str]):
        string_array = np.array(labels).reshape(-1, 1)
        return torch.Tensor(self.ordinal_encoder.fit_transform(string_array))

    def __repr__(self):
        _base = super().__repr__()
        categories = ", ".join(self.ordinal_encoder.categories[0])
        return _base.format(categories)


class MultiLabelEncode(Transform):
    """Transforms a sequence of labels into a binary vector.

    Attributes:
        label_binarizer: the wrapped binarizer instance

    Raises:
        ValueError: if any of the labels are not found in the encoder classes
    """

    label_binarizer: MultiLabelBinarizer

    def __init__(self, classes: Collection[str]):
        self.label_binarizer = MultiLabelBinarizer(
            classes=list(classes), sparse_output=False
        )

    @functools.cached_property
    def classes(self):
        return set(self.label_binarizer.classes)

    def __call__(self, labels: ExampleMultiLabel | BatchMultiLabel) -> torch.Tensor:
        # If labels is a single example, convert it to a batch
        if not labels or isinstance(labels[0], str):
            labels = [labels]
        self._check_inputs(labels)
        encoded = self.label_binarizer.fit_transform(labels)
        return torch.from_numpy(encoded)

    def _check_inputs(self, labels: BatchMultiLabel):
        unique_labels = set(itertools.chain.from_iterable(labels))
        if not unique_labels <= self.classes:
            missing_labels = unique_labels - self.classes
            raise ValueError(
                f"Labels {missing_labels} not found in the encoder classes {self.classes}"
            )

    def __repr__(self):
        _base = super().__repr__()
        classes = ", ".join(self.label_binarizer.classes)
        return _base.format(classes)
