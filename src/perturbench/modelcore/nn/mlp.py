import torch.nn as nn
from torch.distributions import RelaxedBernoulli


def gumbel_softmax_bernoulli(probs, temperature=0.5):
    # Create a RelaxedBernoulli distribution with the specified temperature
    relaxed_bernoulli = RelaxedBernoulli(temperature, probs=probs)

    # Sample from the relaxed distribution
    soft_sample = relaxed_bernoulli.rsample()

    # Quantize the soft sample to get a hard sample
    hard_sample = (soft_sample > 0.5).float()

    # Use straight-through estimator
    st_sample = hard_sample - soft_sample.detach() + soft_sample

    return st_sample


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float | None = None,
        norm: str | None = "layer",
        elementwise_affine: bool = False,
    ):
        """Class for defining MLP with arbitrary number of layers"""
        super(MLP, self).__init__()

        if norm not in ["layer", "batch", None]:
            raise ValueError("norm must be one of ['layer', 'batch', None]")

        layers = nn.Sequential()
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(0, n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == "layer":
                layers.append(
                    nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)
                )
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = layers

    def forward(self, x):
        return self.network(x)


class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float | None = None,
        norm: str | None = "layer",
        elementwise_affine: bool = False,
    ):
        super(ResMLP, self).__init__()

        layers = nn.Sequential()
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(0, n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.layers = layers

    def forward(self, x):
        return x + self.layers(x)


class MaskNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float | None = None,
        norm: str | None = "layer",
        elementwise_affine: bool = False,
    ):
        """
        Implements a mask module. Similar to a standard MLP, but the output will be discrete 0's and 1's
        and gradients are calculated with the Gumbel-Softmax relaxation.
        """
        super(MaskNet, self).__init__()

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            norm=norm,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x):
        m_probs = self.mlp(x).sigmoid()
        m = gumbel_softmax_bernoulli(m_probs)
        return m
