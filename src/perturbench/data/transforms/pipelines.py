from .base import Dispatch, Compose
from .encoders import OneHotEncode, MultiLabelEncode
from .ops import ToDense, ToFloat, MapApply, Unsqueeze


class LinearModelPipeline(Dispatch):
    """Linear model transform pipeline."""

    def __init__(
            self,
            perturbation_uniques: set[str],
            covariate_uniques: dict[str, set],
            use_perturbation_embedding: bool = False,
    ) -> None:
        # Set up covariates transform
        # For continuous covariates (uniques is None), use ToFloat + Unsqueeze to get [batch, 1]
        # For categorical covariates, use OneHotEncode + ToFloat to get [batch, num_classes]
        covariate_transform = {
            key: Compose([ToFloat(), Unsqueeze(dim=-1)]) if uniques is None else Compose(
                [OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        if use_perturbation_embedding:
            perturbation_transform = Compose(
                [
                    ToDense(),
                    ToFloat()
                ]
            )
        else:
            perturbation_transform = Compose(
                [
                    MultiLabelEncode(perturbation_uniques),
                    ToFloat(),
                ]
            )

        # Initialize the pipeline
        super().__init__(
            perturbations=perturbation_transform,
            gene_expression=Compose([ToDense(), ToFloat()]),
            covariates=MapApply(covariate_transform),
        )


class LinearModelPipelineControls(Dispatch):
    """Linear model transform pipeline."""

    def __init__(
            self,
            perturbation_uniques: set[str],
            covariate_uniques: dict[str, set],
            use_perturbation_embedding: bool = False,
    ) -> None:
        # Set up covariates transform
        # For continuous covariates (uniques is None), use ToFloat + Unsqueeze to get [batch, 1]
        # For categorical covariates, use OneHotEncode + ToFloat to get [batch, num_classes]
        covariate_transform = {
            key: Compose([ToFloat(), Unsqueeze(dim=-1)]) if uniques is None else Compose(
                [OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        if use_perturbation_embedding:
            perturbation_transform = Compose(
                [
                    ToDense(),
                    ToFloat()
                ]
            )
        else:
            perturbation_transform = Compose(
                [
                    MultiLabelEncode(perturbation_uniques),
                    ToFloat(),
                ]
            )

        # Initialize the pipeline
        super().__init__(
            perturbations=perturbation_transform,
            gene_expression=Compose([ToDense(), ToFloat()]),
            covariates=MapApply(covariate_transform),
            controls=Compose([ToDense(), ToFloat()]),
        )
