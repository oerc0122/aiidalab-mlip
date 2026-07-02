"""Defines the main AiiDAlab MLIP application page."""
from datetime import datetime

import aiidalab_widgets_base as awb
import ipywidgets as ipw
from aiida_mlip.calculations.geomopt import GeomOpt
from alc_aiidalab_widgets.widgets import Status
from IPython.display import display

from aiidalab_mlip.common.navigation import QuickAccessButtons
from aiidalab_mlip.models.main import MainAppModel
from aiidalab_mlip.steps import (
    ModelWizardStep,
    ResultsWizardStep,
    RunWizardStep,
    # TrainingWizardStep,
    StructureWizardStep,
    TaskWizardStep,
)


class MainApp:
    """The main AiiDAlab MLIP application class."""

    def __init__(self) -> None:
        """MainApp constructor."""
        self.model = MainAppModel()
        self.view = MainAppView(self.model)
        display(self.view)


class MainAppView(ipw.VBox):
    """The main app view."""

    def __init__(self, model: MainAppModel, **kwargs) -> None:
        """MainAppView constructor."""
        logo = ipw.HTML(
            """
            <div class="app-container logo" style="text-align: center;">
                <h1> Machine Learning Interatomic Potentials</h1>
            </div>
            """,
            layout={"margin": "auto"},
        )

        subtitle = ipw.HTML(
            """
            <h2 style="text-align: center;">
                Train and deploy ML potentials for molecular simulations
            </h2>
            """
        )

        nav_btns = QuickAccessButtons()

        header = ipw.VBox(
            children=[
                logo,
                subtitle,
            ],
            layout={"margin": "auto"},
        )

        footer = ipw.HTML(
            f"""
            <footer style="text-align: center; margin-top: 20px;">
                Copyright (c) {datetime.now().year} MLIP Development Team
            </footer>
            """,
        )

        self.main = WizardWidget(model)

        super().__init__(layout={}, children=[header, nav_btns, self.main, footer], **kwargs)


class WizardWidget(ipw.VBox):
    """Widget to hold the main MLIP application wizard."""

    def __init__(self, model: MainAppModel, **kwargs) -> None:
        """
        WizardWidget constructor.

        Parameters
        ----------
        model : MainAppModel
            The application data model
        **kwargs :
            Keyword arguments passed to ipywidgets.VBox.__init__()
        """
        self.structure_step = StructureWizardStep(model.structure_model)
        self.model_step = ModelWizardStep(model.mlip_model)
        self.task_step = TaskWizardStep(model.task_model)
        self.run_step = RunWizardStep(model)
        self.results_step = ResultsWizardStep(model.results_model)

        # Link structure to prediction step
        # def update_prediction_structure(change: dict[str, Atoms]) -> None:
        #     self.prediction_step._parent_structure = change["new"]

        # model.structure_model.observe(update_prediction_structure, names="structure")

        self._wizard_app_widget = awb.WizardAppWidget(
            steps=[
                ("Select Structure", self.structure_step),
                ("Select model", self.model_step),
                ("Select task", self.task_step),
                # ("Train MLIP", self.training_step),
                ("Run", self.run_step),
                ("View Results", self.results_step),
            ],
        )

        self.results_step.disabled = True

        super().__init__(
            children=[self._wizard_app_widget],
            **kwargs,
        )


class Run(ipw.VBox, awb.WizardAppWidgetStep):
    def __init__(self, model: MainAppModel, **kwargs):
        """
        Initialize prediction wizard step.

        Parameters
        ----------
        model : MainAppModel
            The model.
        """
        self.model = model

        self.title = ipw.HTML("<h3>Run calculation</h3>")

        self.info = ipw.HTML(
            """
            <p>Run calculations using the trained MLIP model.</p>
            """
        )

        self.run_button = ipw.Button(
            description="Run Calculation", button_style="success", disabled=True
        )
        self.run_button.on_click(self._on_run_click)

        self.status = Status()
        self.logspace = ipw.Output()

        super().__init__(
            children=[
                self.title,
                self.info,
                self.run_button,
                self.status,
                self.logspace,
            ],
            **kwargs,
        )

    def _on_run_click(self, button):
        """Handle run button click."""
        with self.logspace:
            self.logspace.clear_output()

            # Get structure from Step 1 (passed via main.py observer)
            structure = self.model.structure_model.structure

            if structure is None:
                print("Error: No structure uploaded. Please go to Step 1 and upload a structure.")
                self.status.failure("No structure available.")
                return

            calc_type = self.model.task_model.task

            code = self.model.mlip_model.code
            model = self.model.mlip_model.model

            # Build calculation based on type
            match calc_type:
                case "geometry_opt":
                    print("Setting up Geometry Optimization...")
                    builder = GeomOpt.get_builder()
                    builder.code = code
                    builder.struct = structure_node
                    builder.model = model
                    builder.arch = orm.Str(model.architecture)
                    builder.device = orm.Str("cpu")
                    builder.fmax = orm.Float(0.05)
                    builder.steps = orm.Int(500)
                    display_name = "Geometry Optimization"

                case "single_point":
                    print("Setting up Single Point calculation...")
                    builder = Singlepoint.get_builder()
                    builder.code = code
                    builder.struct = structure_node
                    builder.model = model
                    builder.arch = orm.Str(model.architecture)
                    builder.device = orm.Str("cpu")
                    display_name = "Single Point"

                case "md":
                    print("Error: MD calculations not yet implemented")
                    self.status.value = "<p style='color: orange;'>Warning: MD coming soon</p>"
                    return

            # Set computation resources
            builder.metadata.options.resources = {"num_machines": 1}
            builder.metadata.options.max_wallclock_seconds = 3600

            # Submit calculation
            print(f"Submitting {display_name}...")
            node = engine.submit(builder)

            print(f"Submitted successfully!")
            print(f"  PK: {node.pk}")
            print(f"  UUID: {node.uuid}")
            print(f"\nCheck status with: verdi process list")
            print(f"Or view results in Step 4 with PK: {node.pk}")

            self.status.value = (
                f"<p style='color: green;'>{display_name} submitted (PK: {node.pk})</p>"
            )
            self.model.calculation_node = node

            # Store PK for results step (if main app model is available)
            if hasattr(self, "_app_model") and hasattr(self._app_model, "results_model"):
                self._app_model.results_model.calculation_pk = node.pk
