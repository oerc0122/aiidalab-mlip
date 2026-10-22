"""Defines the data distribution step."""

from alc_aiidalab_widgets.widgets.dependent_sliders import ProportionSlider
from collections.abc import Generator
from typing import ClassVar

import ipywidgets as ipw
from aiida_mlip.calculations.descriptors import Descriptors
from alc_aiidalab_widgets.layouts import WizardStep
from plotly.graph_objects import FigureWidget


class DistributeWizardStep(WizardStep):
    """Wizard step for data distribution."""

    DEFAULTS: ClassVar[dict[str, float]] = {"Train": 60, "Test": 30, "Validation": 10}

    def __init__(self, model, **kwargs):

        self.proportion = ProportionSlider[float](
            ipw.FloatSlider(description=label, min=0, max=100, value=initial, readout=False)
            for label, initial in self.DEFAULTS.items()
        )
        self.plot = FigureWidget()
        self.options = ipw.MultiSelect(
            options=("invariants_only", "calc_per_element", "calc_per_atom"),
        )
        self.compute = ipw.Button(description="Calculate descriptors")
        self.compute.observe(self._compute_descriptors)

        super().__init__(
            title="Run Predictions",
            info="Run calculations using the trained MLIP model.",
            widgets=(self.proportion, self.options, self.compute, self.plot),
            **kwargs,
        )

    def _compute_descriptors(self, *_):
        builder = Descriptors.get_builder()
        builder.invariants_only = "invariants_only" in self.options.value
        builder.calc_per_element = "calc_per_element" in self.options.value
        builder.calc_per_atom = "calc_per_atom" in self.options.value
        builder.code = self.model.task_model.code
        builder.struct = structure
        builder.model = model
        builder.arch = orm.Str(self.model.mlip_model.arch)
        builder.device = orm.Str(self.model.mlip_model.device)
