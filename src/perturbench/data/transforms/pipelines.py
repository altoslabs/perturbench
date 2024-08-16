from .base import Dispatch, Compose
from .encoders import OneHotEncode, MultiLabelEncode
from .ops import ToDense, ToFloat, MapApply


class SingleCellPipeline(Dispatch):
    """Single cell transform pipeline."""

    def __init__(
        self,
        perturbation_uniques: set[str],
        covariate_uniques: dict[str:set],
    ) -> None:
        # Set up covariates transform
        covariate_transform = {
            key: Compose([OneHotEncode(uniques), ToFloat()])
            for key, uniques in covariate_uniques.items()
        }
        # Initialize the pipeline
        super().__init__(
            perturbations=Compose(
                [
                    MultiLabelEncode(perturbation_uniques),
                    ToFloat(),
                ]
            ),
            gene_expression=ToDense(),
            covariates=MapApply(covariate_transform),
        )
