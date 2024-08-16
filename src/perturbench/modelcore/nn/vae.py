import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import Tuple
from .mlp import MLP


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(LogisticRegression, self).__init__()

        self.weights = nn.Parameter(torch.zeros(input_dim))
        self.biases = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.weights * x + self.biases
        # outputs = torch.sigmoid(x)

        return x


class ConditionalGaussian(nn.Module):
    def __init__(self, condition_dim: int) -> None:
        super(ConditionalGaussian, self).__init__()
        self.mu_layer_true = nn.Parameter(torch.zeros(condition_dim))
        self.mu_layer_false = nn.Parameter(torch.zeros(condition_dim))
        self.scale_true = nn.Parameter(torch.ones(condition_dim))
        self.scale_false = nn.Parameter(torch.ones(condition_dim))

    def forward(self, condition) -> Tuple[torch.Tensor, torch.Tensor]:

        mu = condition * self.mu_layer_true + (1 - condition) * self.mu_layer_false
        scale = condition * self.scale_true + (1 - condition) * self.scale_false
        # dist = Normal(mu, torch.exp(0.5 * scale)) #F.softplus(scale).clip(min=1e-3)

        return mu, scale


class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 2,
    ) -> None:
        """
        BaseEncoder for perturbation model.
        """

        super(BaseEncoder, self).__init__()

        self.mlp = MLP(
            input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, n_layers=n_layers
        )

        self.mu_x = nn.Linear(hidden_dim, latent_dim)
        self.var_x = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.mlp(x)
        z_mu = self.mu_x(x)
        z_log_var = self.var_x(x)

        return z_mu, z_log_var


class PerturbationConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim_x: int,
        hidden_dim_cond: int,
        output_dim: int,
        n_layers: int = 2,
    ) -> None:
        """
        Encoder for perturbation model. The model contains two separate neural networks for perturbation indicators
        and expression data. This is crucial to get this model to work it seems.
        """

        super(PerturbationConditionalEncoder, self).__init__()

        self.expression_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim_x,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )
        self.condition_mlp = MLP(
            input_dim=condition_dim,
            hidden_dim=hidden_dim_cond,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )

        self.mu_x = nn.Linear(hidden_dim_x, output_dim)
        self.var_x = nn.Linear(hidden_dim_x, output_dim)

    def forward(self, x: torch.Tensor, condition: torch.tensor) -> torch.Tensor:

        x = self.expression_mlp(x)
        condition = self.condition_mlp(condition)
        x = x + condition
        z_mu = self.mu_x(x)
        z_log_var = self.var_x(x)

        return (
            z_mu,
            z_log_var,
        )  # torch.cat([mu_condition, z_mu], dim=-1), torch.cat([var_condition, z_log_var], dim=-1)


class ResidualPerturbationConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim_x: int,
        hidden_dim_cond: int,
        n_layers: int = 2,
    ) -> None:
        """
        Encoder for perturbation model. The model contains two separate neural networks for perturbation indicators
        and expression data. This is crucial to get this model to work it seems.
        """

        super(ResidualPerturbationConditionalEncoder, self).__init__()

        self.expression_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim_x,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )
        self.condition_mlp = MLP(
            input_dim=condition_dim,
            hidden_dim=hidden_dim_cond,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )

        self.mu_x = nn.Linear(hidden_dim_x, input_dim)
        self.var_x = nn.Linear(hidden_dim_x, input_dim)

    def forward(self, x: torch.Tensor, condition: torch.tensor) -> torch.Tensor:

        z = self.expression_mlp(x)
        condition = self.condition_mlp(condition)
        z = z + condition
        z_mu = x + self.mu_x(z)
        z_log_var = self.var_x(z)

        return (
            z_mu,
            z_log_var,
        )  # torch.cat([mu_condition, z_mu], dim=-1), torch.cat([var_condition, z_log_var], dim=-1)


class PerturbationSplitEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim_x: int,
        hidden_dim_cond: int,
        output_dim: int,
        n_layers: int = 2,
    ) -> None:
        """
        Encoder for perturbation model.
        """

        super(PerturbationSplitEncoder, self).__init__()

        self.expression_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim_x,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )
        self.condition_mlp = MLP(
            input_dim=condition_dim,
            hidden_dim=hidden_dim_cond,
            output_dim=hidden_dim_x,
            n_layers=n_layers,
        )
        self.mu_condition = nn.Linear(hidden_dim_x, condition_dim)
        self.var_condition = nn.Linear(hidden_dim_x, condition_dim)
        self.mu_x = nn.Linear(hidden_dim_x, output_dim - condition_dim)
        self.var_x = nn.Linear(hidden_dim_x, output_dim - condition_dim)

    def forward(self, x: torch.Tensor, condition: torch.tensor) -> torch.Tensor:

        # expression features only block (for basal state)
        x = self.expression_mlp(x)
        z_mu = self.mu_x(x)
        z_log_var = self.var_x(x)

        # mixed features block (for perturbation state)
        condition = self.condition_mlp(condition)
        condition = x + condition
        mu_condition = self.mu_condition(condition)
        var_condition = self.var_condition(condition)

        return torch.cat([mu_condition, z_mu], dim=-1), torch.cat(
            [var_condition, z_log_var], dim=-1
        )


class VariationalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float | None = None,
    ) -> None:
        """
        Encoder for perturbation model. The model contains two separate neural networks for perturbation indicators
        and expression data. This is crucial to get this model to work it seems.
        """

        super(VariationalEncoder, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.mu_x = nn.Linear(hidden_dim, latent_dim)
        self.var_x = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.network(x)

        z_mu = self.mu_x(x)
        z_var = self.var_x(x)
        dist = Normal(z_mu, torch.exp(z_var).sqrt())
        latent = dist.rsample()

        return dist, latent


class VariationalDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float | None = None,
    ) -> None:
        super(VariationalDecoder, self).__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.mu_z = nn.Linear(hidden_dim, output_dim)
        self.var_z = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        z = self.network(z)

        x_mu = self.mu_z(z)
        x_log_var = self.var_z(z)
        x_var = torch.exp(x_log_var)

        return x_mu, x_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(Decoder, self).__init__()

        # Reverse transformation from the Encoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, z: torch.Tensor):
        z = F.relu(self.bn1(self.fc1(z)))
        # z = self.dropout(z)
        z = F.relu(self.bn2(self.fc2(z)))
        # z = self.dropout(z)
        # z = F.relu(self.fc1(z))
        # z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        # Reconstruction
        prediction = F.relu(self.out(z))

        return prediction
