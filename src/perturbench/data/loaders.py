from torch.utils.data import (
    Dataset,
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    DistributedSampler,
)
import pandas as pd
from pytorch_metric_learning.samplers.m_per_class_sampler import MPerClassSampler
from torch.utils.data.distributed import _T_co
from typing import Optional, Sequence, Iterator
import torch
import math
from .collate import noop_collate
from .utils import build_covariate_to_weight_dict, combine_perturbations


class DistributedWeightedSampler(DistributedSampler):

    def __init__(
        self,
        weights: Sequence[float],
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        replacement: bool = False,
    ) -> None:
        assert len(weights) == len(dataset)
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=True,
                         seed=seed, drop_last=drop_last)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )
        self.weights = weights_tensor
        self.replacement = replacement

    def __iter__(self) -> Iterator[_T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        ).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class BatchedDataLoader(DataLoader):
    """High performance batched DataLoader to access the datasets in batches.

    This dataloader accesses the dataset in batches via __getitems__ interface.
    Typically, a dataset would implement __getitems__ to return a batch of examples
    at once when sampling a batch is faster than sampling a sequence of single
    examples.

    Attributes:
        dataset: the dataset to load the data from
        batch_size: the size of the batch
        shuffle: whether to shuffle the data
        drop_last: whether to drop the last batch if it is smaller than batch_size
        kwargs: additional arguments to pass to DataLoader
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        oversample: bool = False,
        oversample_root: float = 2.0,
        use_m_per_class_sampler: bool = False,
        num_samples_per_condition: int = 1000,
        ddp_mode: bool = False,
        **kwargs,
    ):
        if ddp_mode:
            if oversample:
                covariates = dataset.covariates.copy()
                covariates.update({'perturbations': ['-'.join(pert) for pert in dataset.perturbations]})
                weights_dictionary = build_covariate_to_weight_dict(pd.DataFrame(covariates), oversample_root)

                weights = []
                for i in range(0, len(dataset)):
                    cov_values = [values[i] for values in dataset.covariates.values()]
                    cov_key = '_'.join(cov_values)
                    pert = '-'.join(dataset.perturbations[i])
                    weight = weights_dictionary[cov_key + '_' + pert]
                    weights.append(weight)

                sampler = DistributedWeightedSampler(weights, dataset, replacement=True)
            else:
                sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if use_m_per_class_sampler:
                condition_df = pd.DataFrame(dataset.covariates)
                condition_df.insert(0, 'perturbation', combine_perturbations(dataset.perturbations, ','))
                labels = condition_df.apply('_'.join, axis=1).values
                sampler = MPerClassSampler(labels, m=num_samples_per_condition, length_before_new_iter=len(labels))
            elif shuffle:
                if oversample:
                    assert oversample_root > 0, "Oversample root must be greater than 0"

                    weights_dictionary = build_covariate_to_weight_dict(pd.DataFrame(dataset.covariates), oversample_root)
                    weights = []
                    for i in range(0, len(dataset)):
                        cov_values = [values[i] for values in dataset.covariates.values()]
                        cov_key = '_'.join(cov_values)
                        weight = weights_dictionary[cov_key]
                        weights.append(weight)

                    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

                else:
                    sampler = RandomSampler(dataset)

            else:
                sampler = SequentialSampler(dataset)

        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last
        )

        if 'batch_sampler' in kwargs:
            kwargs.pop('batch_sampler')
        if 'collate_fn' in kwargs:
            kwargs.pop('collate_fn')

        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=noop_collate(),
            **kwargs
        )
