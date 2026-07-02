"""Model for task step."""

from traitlets import Dict, Unicode

from .base import Model


class TaskModel(Model):
    """Model for task step."""

    task = Unicode()
    task_parameters = Dict(key_trait=Unicode())
