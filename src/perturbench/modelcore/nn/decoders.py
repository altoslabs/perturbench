import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from .mlp import MLP
from typing import Literal
from .utils import ZeroInflatedNegativeBinomial
from abc import abstractmethod


class Decoder(nn.Module):
    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def reconstruction_loss(predictions, target):
        pass


class DeepGaussian(Decoder):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout) -> None:
        super(DeepGaussian, self).__init__()
        # Define a simple feedforward network
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)
        self.network = MLP(input_dim, hidden_dim, hidden_dim, n_layers, dropout)
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        # Forward pass through the network to get mean and log(std) for each dimension
        x = self.network(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return dist.Normal(mean, torch.exp(log_var * 0.5).clip(min=1e-8))

    @staticmethod
    def reconstruction_loss(
        predictions: dist.Distribution, target: torch.Tensor, reduction: str = "mean"
    ):
        if reduction == "mean":
            return -predictions.log_prob(target).sum(-1).mean()
        elif reduction == "none":
            return -predictions.log_prob(target).sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")


class DeepIsotropicGaussian(Decoder):
    # zyan: todo, isotropic means the covariance matrix is diagonal. Here, it means not learning variance at all.
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        softplus_output: bool = False,
    ) -> None:
        super(DeepIsotropicGaussian, self).__init__()
        self.network = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)
        self.softplus_output = softplus_output

    def forward(self, x, **kwargs):
        # Forward pass through the network to get the predicted values
        if self.softplus_output:
            predictions = F.softplus(self.network(x))
        else:
            predictions = self.network(x)
        return predictions

    @staticmethod
    def reconstruction_loss(
        predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ):
        if reduction == "mean":
            return F.mse_loss(predictions, target, reduction="none").sum(-1).mean()
        elif reduction == "none":
            return F.mse_loss(predictions, target, reduction="none").sum(-1)
        elif reduction == 'featurewise_mean':
            return F.mse_loss(predictions, target, reduction="none").mean()
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")


class DeepPoisson(Decoder):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        library_size: Literal["observed", "learned"] | None = "observed",
    ) -> None:
        super(DeepPoisson, self).__init__()
        self.library_size = library_size
        self.rho = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)

        if self.library_size == "learned":
            self.library_size_net = MLP(input_dim, hidden_dim, 1, n_layers, dropout)

    def forward(self, x, library_size=None):
        rho = self.rho(x)
        rho = torch.softmax(rho, dim=-1)

        if self.library_size == "observed" and library_size is None:
            raise ValueError(
                "Library size must be provided if library size is observed"
            )
        elif self.library_size == "observed" and library_size is not None:
            lib = library_size.reshape(-1, 1)
        elif self.library_size == "learned":
            lib = F.relu(self.library_size_net(x)).clip(min=1e-5).reshape(-1, 1) + 1_000
        else:
            raise ValueError("Missing library_size argument")

        rate = lib.expand(*rho.shape).clip(min=1e-5) * rho
        return dist.poisson.Poisson(rate=rate)

    @staticmethod
    def reconstruction_loss(
        predictions: dist.Distribution, target: torch.Tensor, reduction: str = "mean"
    ):
        if reduction == "mean":
            return -predictions.log_prob(target).sum(-1).mean()
        elif reduction == "none":
            return -predictions.log_prob(target).sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")


class DeepPoissonGamma(Decoder):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        library_size: Literal["observed", "learned"] | None = "observed",
        use_legacy_negative_binomial: bool = False,
    ) -> None:
        super(DeepPoissonGamma, self).__init__()

        self.library_size = library_size
        self.rho = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)
        self.log_theta = nn.Parameter(
            torch.randn(output_dim) * 1 / torch.sqrt(torch.tensor(output_dim))
        )

        if self.library_size == "learned":
            self.library_size_net = MLP(input_dim, hidden_dim, 1, n_layers, dropout)

        self.use_legacy_negative_binomial = use_legacy_negative_binomial

    def forward(self, x, library_size=None):
        rho = self.rho(x)
        rho = torch.softmax(rho, dim=-1)
        if self.library_size == "observed" and library_size is None:
            raise ValueError(
                "Library size must be provided if library size is observed"
            )
        elif self.library_size == "observed" and library_size is not None:
            lib = library_size.reshape(-1, 1)
        elif self.library_size == "learned":
            lib = F.relu(self.library_size_net(x)).clip(min=1e-5).reshape(-1, 1) + 1_000
        else:
            raise ValueError("Missing library_size argument")

        if self.use_legacy_negative_binomial:
            concentration = lib.expand(*rho.shape).clip(min=1e-5) * rho
            return dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=self.log_theta.expand(x.shape[0], self.log_theta.shape[0]),
                validate_args=False,
            )
        else:
            mean = (lib.expand(*rho.shape) * rho).clip(min=1e-5)
            concentration = torch.exp(self.log_theta).unsqueeze(0).repeat(x.shape[0], 1)
            return dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=torch.log(mean / (concentration + 1e-5)),
                validate_args=False,
            )

    @staticmethod
    def reconstruction_loss(
        predictions: dist.Distribution, target: torch.Tensor, reduction: str = "mean"
    ):
        if reduction == "mean":
            return -predictions.log_prob(target).sum(-1).mean()
        elif reduction == "none":
            return -predictions.log_prob(target).sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")


class ZeroInflatedPoissonGamma(Decoder):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        library_size: Literal["observed", "learned"] | None = "observed",
        use_legacy_negative_binomial: bool = False,
        dispersion_by_gene_cell: bool = False,
    ) -> None:
        super(ZeroInflatedPoissonGamma, self).__init__()

        self.library_size = library_size
        self.rho = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)
        self.mask_logits = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)

        self.dispersion_by_gene_cell = dispersion_by_gene_cell
        if self.dispersion_by_gene_cell:
            self.log_theta_mlp = MLP(
                input_dim, hidden_dim, output_dim, n_layers, dropout
            )
        else:
            self.log_theta = nn.Parameter(
                torch.randn(output_dim) * 1 / torch.sqrt(torch.tensor(output_dim))
            )

        if self.library_size == "learned":
            self.library_size_net = MLP(input_dim, hidden_dim, 1, n_layers, dropout)

        self.use_legacy_negative_binomial = use_legacy_negative_binomial

    def forward(self, x, library_size=None):
        rho = self.rho(x)
        zero_prob_logits = self.mask_logits(x)
        rho = torch.softmax(rho, dim=-1)
        if self.library_size == "observed" and library_size is None:
            raise ValueError(
                "Library size must be provided if library size is observed"
            )
        elif self.library_size == "observed" and library_size is not None:
            lib = library_size.reshape(-1, 1)
        elif self.library_size == "learned":
            lib = F.relu(self.library_size_net(x)).clip(min=1e-5).reshape(-1, 1) + 1_000
        else:
            raise ValueError("Missing library_size argument")

        if self.dispersion_by_gene_cell:
            log_theta = self.log_theta_mlp(x)
        else:
            log_theta = self.log_theta

        if self.use_legacy_negative_binomial:
            concentration = lib.expand(*rho.shape).clip(min=1e-5) * rho
            if log_theta.ndim == 1:
                log_theta = log_theta.expand(x.shape[0], log_theta.shape[0])
            negative_binomial_dist = dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=log_theta,
                validate_args=False,
            )
        else:
            mean = (lib.expand(*rho.shape) * rho).clip(min=1e-5)
            concentration = torch.exp(log_theta)
            if concentration.ndim == 1:
                concentration = concentration.unsqueeze(0).repeat(x.shape[0], 1)
            # NB is typically for count-based data. Nevertheless, it is straightforward to extend to fractional counts,
            # i.e. continuous data: simply setting `validate_args=False` for this purpose.
            # The traditional mean and variance formulas are good approximation in the continuous case.
            # See more in: https://stats.stackexchange.com/questions/310676/continuous-generalization-of-the-negative-binomial-distribution
            negative_binomial_dist = dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=torch.log(mean / (concentration + 1e-5)),
                validate_args=False,
            )

        return ZeroInflatedNegativeBinomial(negative_binomial_dist, zero_prob_logits)

    @staticmethod
    def reconstruction_loss(
        predictions: dist.Distribution, target: torch.Tensor, reduction: str = "mean"
    ):
        if reduction == "mean":
            return -predictions.log_prob(target).sum(-1).mean()
        elif reduction == "none":
            return -predictions.log_prob(target).sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")
