"""Results viewing wizard step."""
import traceback

import aiidalab_widgets_base as awb
import ase
import ipywidgets as ipw
import nglview
import numpy as np
from aiida import orm
from aiida.orm import ProcessNode, QueryBuilder
from alc_aiidalab_widgets.layouts import Step
from ase import Atoms
from IPython.display import display
from ipywidgets import Button


class ResultsWizardStep(Step, awb.WizardAppWidgetStep):
    """Wizard step for viewing results."""

    def __init__(self, model, **kwargs):
        """
        Initialize results wizard step.

        Parameters
        ----------
        model : ResultsModel
            The results data model
        """
        self.model = model

        # Process list refresh button
        self.refresh_button = ipw.Button(
            description="Refresh Process List",
            button_style="primary",
            icon="refresh",
            layout={"margin": "auto", "width": "60%"},
        )
        self.refresh_button.on_click(self._on_refresh_click)

        # Process selector dropdown
        self.process_selector = ipw.Dropdown(
            options=[],
            description="Select:",
            disabled=False,
            layout={"margin": "auto", "width": "60%"},
        )
        self.process_selector.observe(self._on_process_select, names="value")

        # Manual PK input (alternative to dropdown)
        self.pk_input = ipw.IntText(
            value=0,
            description="Or enter PK:",
            disabled=False,
            layout={"margin": "auto", "width": "60%"},
        )

        self.load_button = ipw.Button(
            description="Load Results",
            button_style="info",
            layout={"margin": "auto", "width": "20%"},
        )
        self.load_button.on_click(self._on_load_click)

        self.process_list_area = ipw.Output(layout={"margin": "auto", "width": "100%"})

        # Auto-refresh process list on load
        self._refresh_process_list()

        super().__init__(
            title="View Results",
            info="View and analyze calculation results.",
            widgets=[
                self.refresh_button,
                self.process_list_area,
                ipw.HBox([self.process_selector]),
                ipw.HBox([self.pk_input, self.load_button]),
            ],
            submittable=False,
            **kwargs,
        )

        # Observe calculation_pk changes to auto-load
        self.model.observe(self._on_pk_change, names="calculation_pk")

    def _on_refresh_click(self, _) -> None:
        """Refresh the process list."""
        self._refresh_process_list()

    def _refresh_process_list(self) -> None:
        """Load and display recent processes."""
        with self.process_list_area:
            self.process_list_area.clear_output()

            try:
                # Query recent calculations
                qb = QueryBuilder()
                qb.append(
                    ProcessNode,
                    filters={"attributes.process_label": {"in": ["Singlepoint", "GeomOpt", "MD"]}},
                    project=[
                        "id",
                        "ctime",
                        "attributes.process_label",
                        "attributes.process_state",
                        "attributes.exit_status",
                    ],
                )
                qb.order_by({ProcessNode: {"ctime": "desc"}})
                qb.limit(20)

                results = qb.all()

                if not results:
                    print("No MLIP calculations found. Submit one in Step 4!")
                    self.process_selector.options = []
                    return

                print("Recent Calculations:\n")
                print(f"{'PK':<6} {'Type':<15} {'State':<12} {'Exit':<6} {'Created'}")
                print("-" * 70)

                options = []
                for pk, ctime, label, state, exit_status in results:
                    status_icon = "OK" if exit_status == 0 else "ERR" if exit_status else "RUN"
                    time_str = ctime.strftime("%Y-%m-%d %H:%M")
                    print(f"{pk:<6} {label:<15} {state or 'N/A':<12} {status_icon:<6} {time_str}")

                    # Add to dropdown options
                    display = f"PK {pk} - {label} ({state or 'N/A'}) - {time_str}"
                    options.append((display, pk))

                self.process_selector.options = options
                if options:
                    self.process_selector.value = options[0][1]  # Select most recent

            except Exception as e:
                print(f"Error loading processes: {e}")

                traceback.print_exc()

    def _on_process_select(self, change):
        """Handle process selection from dropdown."""
        if change["new"]:
            self.pk_input.value = change["new"]

    def _on_pk_change(self, change):
        """Auto-load results when PK is set."""
        if change["new"] > 0:
            self.pk_input.value = change["new"]
            self._on_load_click(None)

    def _on_load_click(self, button):
        """Load and display results for the given PK."""
        with self.logspace:
            self.logspace.clear_output()

            pk = self.pk_input.value
            if pk <= 0:
                print("Error: Please enter a valid process PK")
                self.status.failure("Invalid PK")
                return

            try:
                # Load the calculation node
                node = orm.load_node(pk)

                print(f"=== Calculation Results (PK: {pk}) ===\n")
                print(f"Type: {node.process_label}")
                print(f"State: {node.process_state}")
                print(f"Exit status: {node.exit_status}")
                print(f"Created: {node.ctime}")
                print(f"Finished: {node.mtime}\n")

                # Check if calculation finished successfully
                if node.exit_status != 0:
                    print(f"Warning: Calculation exited with status {node.exit_status}")
                    self.status.failure(f"Exit status: {node.exit_status}")
                    return

                # Get outputs
                if "results_dict" not in node.outputs:
                    print("Error: No results_dict output found")
                    self.status.failure("No results available")
                    return

                results = node.outputs.results_dict.get_dict()

                # Display energy
                energy_key = None
                for key in results.get("info", {}).keys():
                    if "energy" in key.lower():
                        energy_key = key
                        break

                if energy_key:
                    energy = results["info"][energy_key]
                    print(f"Energy: {energy:.6f} eV")

                # Display forces info
                force_key = None
                for key in results.keys():
                    if "force" in key.lower():
                        force_key = key
                        break

                if force_key:
                    forces = results[force_key]

                    force_magnitudes = np.linalg.norm(forces, axis=1)
                    print(f"Forces:")
                    print(f"   Max: {force_magnitudes.max():.4f} eV/Å")
                    print(f"   Mean: {force_magnitudes.mean():.4f} eV/Å")
                    print(f"   RMS: {np.sqrt((force_magnitudes**2).mean()):.4f} eV/Å\n")

                # Display stress if available
                stress_key = None
                for key in results.get("info", {}).keys():
                    if "stress" in key.lower():
                        stress_key = key
                        break

                if stress_key:
                    stress = results["info"][stress_key]
                    print(f"Stress tensor: {stress}\n")

                # Display structure info
                n_atoms = len(results["positions"])
                elements = set(results["numbers"])
                print(f"Structure:")
                print(f"   Atoms: {n_atoms}")
                print(f"   Elements: {', '.join(map(str, sorted(elements)))}")
                print(f"   PBC: {results.get('pbc', 'N/A')}\n")

                # Try to visualize structure with nglview
                try:

                    atoms = Atoms(
                        numbers=results["numbers"],
                        positions=results["positions"],
                        cell=results.get("cell"),
                        pbc=results.get("pbc", [True, True, True]),
                    )

                    print("Structure Visualization:")
                    view = nglview.show_ase(atoms)
                    view.add_unitcell()
                    view.add_ball_and_stick()
                    display(view)

                except Exception as e:
                    print(f"Warning: Could not display structure: {e}")

                self.status.value = f"<p style='color: green;'>Loaded results for PK {pk}</p>"

            except Exception as e:
                print(f"Error loading results: {e}")
                traceback.print_exc()
                self.status.value = f"<p style='color: red;'>Error: Failed to load results</p>"
