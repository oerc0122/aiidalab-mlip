"""Model for MLIP training step."""

from traitlets import List, Unicode

from .base import Model


class TrainingModel(Model):
    """Model for MLIP training step."""

    model_type = Unicode(default_value="MACE")
    training_data = List(default_value=[])
