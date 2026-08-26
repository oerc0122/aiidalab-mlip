"""Run calculation."""
import aiidalab_widgets_base as awb
import ipywidgets as ipw
from aiida import engine, orm
from aiida_mlip.calculations.geomopt import GeomOpt
from aiida_mlip.calculations.md import MD
from aiida_mlip.calculations.singlepoint import Singlepoint
from alc_aiidalab_widgets.layouts import Step

from aiidalab_mlip.models import MainAppModel


class RunWizardStep(Step, awb.WizardAppWidgetStep):
    def __init__(self, model: MainAppModel, **kwargs):
        """
        Initialize prediction wizard step.

        Parameters
        ----------
        model : PredictionModel
            The prediction data model
        """
        self.model = model

        self.run_button = ipw.Button(
            description="Run Calculation",
            button_style="success",
            tooltip="Submit the data to the workflow",
            icon="check",
            layout={"margin": "auto", "width": "60%"},
        )
        self.run_button.on_click(self.submit)

        super().__init__(
            title="Run Predictions",
            info="Run calculations using the trained MLIP model.",
            widgets=[self.run_button],
            submittable=False,
            **kwargs,
        )

    def submit(self, _):
        with self.logspace:
            self.logspace.clear_output()

            structure = orm.StructureData(ase=self.model.structure_model.structure)
            if not structure:
                self.status.failure("Error: No structure defined")
                return

            code = self.model.mlip_model.code
            model = self.model.mlip_model.model

            calc_type = self.model.task_model.task
            display_name = self.model.mlip_model.process_label or calc_type
            task_parameters = self.model.task_model.task_parameters

            print(f"Setting up {calc_type} calculation...")
            match calc_type:
                case "Geometry Optimise":
                    builder = GeomOpt.get_builder()

                    builder.opt_cell_fully = orm.Bool(task_parameters["opt_cell_fully"])
                    builder.opt_cell_lengths = orm.Bool(task_parameters["opt_cell_lengths"])
                    builder.fmax = orm.Float(task_parameters["fmax"])
                    builder.steps = orm.Int(task_parameters["steps"])
                    builder.pressure = orm.Float(task_parameters["pressure"])

                case "Single Point":
                    builder = Singlepoint.get_builder()
                    builder.properties = orm.Str(" ".join(task_parameters["properties"]))
                case "Molecular Dynamics":
                    builder = MD.get_builder()
                    builder.ensemble = orm.Str(task_parameters.pop("ensemble"))
                    builder.md_kwargs = orm.Dict(
                        {key: val for key, val in task_parameters.items() if val is not None}
                    )
                case _:
                    raise NotImplementedError(f"Cannot calculate {calc_type}.")

            builder.code = code
            builder.struct = structure
            builder.model = model
            builder.arch = orm.Str(self.model.mlip_model.arch)
            builder.device = orm.Str(self.model.mlip_model.device)

            builder.metadata.options.resources = {
                "num_mpiprocs_per_machine": self.model.mlip_model.ncpus,
                "num_cores_per_machine": self.model.mlip_model.ncpus,
                "num_machines": 1,
                "tot_num_mpiprocs": self.model.mlip_model.ncpus,
            }
            builder.metadata.options.max_wallclock_seconds = 3600

            # Submit calculation
            print(f"Submitting {display_name}...")
            node = engine.submit(builder)
            node.label = self.model.mlip_model.process_label
            node.description = self.model.mlip_model.process_description

            print(f"Submitted successfully!")
            print(f"  PK: {node.pk}")
            print(f"  UUID: {node.uuid}")
            print(f"\nCheck status with: verdi process list")
            print(f"Or view results in Step 4 with PK: {node.pk}")

            self.status.success(f"{display_name} submitted (PK: {node.pk})")

        super().submit(_)
