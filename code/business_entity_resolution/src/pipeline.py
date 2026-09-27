"""
End-to-end Pipeline orchestration module for Business Entity Resolution.
Orchestrates: Data Loading -> Normalization -> Blocking -> Feature Engineering -> Model Inference -> Output Generation.
"""


class EntityResolutionPipeline:
    """
    End-to-end pipeline orchestrating the entire entity resolution solution.
    Placeholder for pipeline integration phase.
    """

    def __init__(self) -> None:
        pass

    def run_training_pipeline(self) -> None:
        """Run full training and validation pipeline."""
        raise NotImplementedError("Pipeline orchestration will be implemented in subsequent phases.")

    def run_inference_pipeline(self) -> None:
        """Run full test inference pipeline."""
        raise NotImplementedError("Pipeline orchestration will be implemented in subsequent phases.")
