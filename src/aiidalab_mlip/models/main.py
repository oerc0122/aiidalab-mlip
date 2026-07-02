"""Main application data model."""

from traitlets import HasTraits

from . import CodeModel, PredictionModel, ResultsModel, StructureModel, TaskModel


class MainAppModel(HasTraits):
    """Main application data model."""

    def __init__(self) -> None:
        """Initialize the main app model."""
        super().__init__()
        self.structure_model = StructureModel()
        self.mlip_model = CodeModel()
        self.task_model = TaskModel()
        self.prediction_model = PredictionModel()
        self.results_model = ResultsModel()
