"""Results viewing wizard step."""

from typing import Any

import ipywidgets as ipw
from aiida.orm import ArrayData, Float, Node, ProcessNode, SinglefileData, StructureData
from aiidalab_widgets_base import ProcessNodesTreeWidget, WizardAppWidgetStep
from aiidalab_widgets_base.loaders import LoadingWidget
from aiidalab_widgets_base.viewers import AIIDA_VIEWER_MAPPING
from alc_aiidalab_widgets.layouts import Step
from alc_aiidalab_widgets.widgets import StructureViewWidget
from traitlets import Instance, observe

# import traceback
# import aiidalab_widgets_base as awb
# import ase
# import ipywidgets as ipw
# import nglview
# import numpy as np
# from aiida import orm
# from aiida.orm import ProcessNode, QueryBuilder

# from ase import Atoms
# from IPython.display import display
# from ipywidgets import Button


class AiidaGradientDataViewWidget(ipw.VBox):
    """Custom widget to display array data produced from ChemShell jobs."""

    def __init__(self, array: ArrayData, **kwargs):
        """AiidaArrayDataViewWidget Constructor.

        Parameters
        ----------
        array : ArrayData
            The AiiDA ArrayData object to display.
        """
        super().__init__(**kwargs)
        self.array = array
        self.array_names = array.get_arraynames()

        self.array_selector = Dropdown(
            options=self.array_names,
            description="Array Label:",
            disabled=False,
            layout={"width": "30%"},
        )
        self._render_array({"new": self.array_selector.index, "old": -1})
        self.array_selector.observe(self._render_array, "index")

        return

    def _render_array(self, change) -> None:
        """Create a HTML table based on the currently selected array."""
        index = change["new"]
        if index == change["old"]:
            return
        values = self.array.get_array(self.array_names[index])
        # Construct HTML Table
        html = "<table style='width:100%; border: 1px solid #ddd; text-align: left; "
        html += "border-collapse: collapse;'>"
        html += "<tr style='background-color: #2196F3; color: white;'>"
        html += "<th>Atom Index</th><th>X</th><th>Y</th><th>Z</th></tr>"

        for idx, row in enumerate(values):
            bg_color = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
            html += f"<tr style='background-color: {bg_color};'>"
            html += f"<td><b>{idx}</b></td><td>{row[0]:.6f}</td><td>{row[1]:.6f}</td>"
            html += f"<td>{row[2]:.6f}</td>"
            html += "</tr>"
        html += "</table>"

        self.children = [self.array_selector, ipw.HTML(html)]


class VibrationalModesViewWidget(ipw.VBox):
    """Custom widget to display vibrational modes produced from ChemShell."""

    def __init__(self, array: ArrayData, **kwargs):
        """VibrationalModesViewWidget Constructor.

        Parameters
        ----------
        array : ArrayData
            The AiiDA ArrayData object to display.
        """
        super().__init__(**kwargs)
        self.array = array
        values = self.array.get_array("Modes")
        # Construct HTML Table
        html = "<table style='width:100%; border: 1px solid #ddd; text-align: left; "
        html += "border-collapse: collapse;'>"
        html += "<tr style='background-color: #2196F3; color: white;'>"
        html += "<th>Mode</th><th>Frequency</th><th>Vib T / K</th><th>ZPE / H</th>"
        html += "</th><th>Energy / H</th></th><th>-TS / H</th></tr>"

        for idx, row in enumerate(values):
            bg_color = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
            html += f"<tr style='background-color: {bg_color};'>"
            html += f"<td><b>{idx}</b></td><td>{row[0]:.6f}</td><td>{row[1]:.6f}</td>"
            html += f"<td>{row[2]:.6f}</td><td>{row[3]:.6f}</td><td>{row[4]:.6f}</td>"
            html += "</tr>"
        html += "</table>"

        self.children = [ipw.HTML(html)]


class CustomAiidaNodeViewWidget(ipw.VBox):
    """
    Custom viewer based on a specific AiiDA node type.

    An extension of the aiida_widgets_base.viewers.AiidaNodeViewWidget
    enabling more customisability when registering viewers with nodes
    returned from ChemShell jobs. The main outline is taken from the base
    aiidalab_widgets_base viewer with an extended viewer() method which
    allows handling of node types which the base viewer has no registered
    visualisation widgets.
    """

    node = Instance(Node, allow_none=True)

    def __init__(self, **kwargs):
        """CustomAiidaNodeViewWidget Constructor."""
        self._output = ipw.Output()
        self.node_views = {}
        self.node_view_loading_message = LoadingWidget("Loading Node View")
        super().__init__(**kwargs)
        self.add_class("aiida-node-view-widget")

    @observe("node")
    def _observe_node(self, change):
        if not ((node := change["new"]) and node != change["old"]):
            return
        if node.uuid in self.node_views:
            self.children = [self.node_views[node.uuid]]
            return
        self.children = [self.node_view_loading_message]
        node_view = self._viewer(node)
        if isinstance(node_view, ipw.DOMWidget):
            self.node_views[node.uuid] = node_view
            self.children = [node_view]
        else:
            with self._output:
                clear_output()
                if change["new"]:
                    display(node_view)
            self.children = [self._output]

    @staticmethod
    def _viewer(node: Node, **kwargs) -> Any:
        """Create a viewer based on the type of Node being visualised."""
        viewer = AIIDA_VIEWER_MAPPING.get(node.node_type)

        match node:
            case ProcessNode():
                # Allow to register specific viewers based on node.process_type
                viewer = AIIDA_VIEWER_MAPPING.get(node.process_type, viewer)
                return viewer(node, **kwargs)
            case SinglefileData():
                # Singlefile data output generally refers to a structure file
                # output from ChemShell jobs
                if "Structure" in node.label:
                    viewer = StructureViewWidget(**kwargs)
                    viewer.assign_structure_from_file(node.filename, node.content)
                    return viewer

            case StructureData():
                viewer = StructureData(**kwargs)
                viewer.assign_structure_from_ase(node.get_ase())
                return viewer

            case ArrayData() if "Energy Derivative" in node.label:
                return AiidaGradientDataViewWidget(node, **kwargs)
            case ArrayData() if "Vibrational" in node.label:
                return VibrationalModesViewWidget(node, **kwargs)

            case Float() if "SCF Energy" in node.label:
                return f"Final SCF Energy (Hartree): {node.value}"

            case _ if viewer:
                return viewer(node, **kwargs)

        # No viewer registered for this type, return node itself
        return node


class ResultsWizardStep(Step, WizardAppWidgetStep):
    """Wizard step for viewing results."""

    def __init__(self, model, **kwargs):
        """
        Initialize results wizard step.

        Parameters
        ----------
        model : ResultsModel
            The results data model
        """
        self.rendered = False
        self.model = model

        # # Process list refresh button
        # self.refresh_button = ipw.Button(
        #     description="Refresh Process List",
        #     button_style="primary",
        #     icon="refresh",
        #     layout={"margin": "auto", "width": "60%"},
        # )
        # self.refresh_button.on_click(self._on_refresh_click)

        # # Process selector dropdown
        # self.process_selector = ipw.Dropdown(
        #     options=[],
        #     description="Select:",
        #     disabled=False,
        #     layout={"margin": "auto", "width": "60%"},
        # )
        # self.process_selector.observe(self._on_process_select, names="value")

        # # Manual PK input (alternative to dropdown)
        # self.pk_input = ipw.IntText(
        #     value=0,
        #     description="Or enter PK:",
        #     disabled=False,
        #     layout={"margin": "auto", "width": "60%"},
        # )

        # self.load_button = ipw.Button(
        #     description="Load Results",
        #     button_style="info",
        #     layout={"margin": "auto", "width": "20%"},
        # )
        # self.load_button.on_click(self._on_load_click)

        # self.process_list_area = ipw.Output(layout={"margin": "auto", "width": "100%"})

        self.update_btn = ipw.Button(
            description="Refresh",
            icon="arrows-rotate",
            disabled=False,
            button_style="info",
            tooltip="Refresh process information.",
            layout={"margin": "auto", "width": "70%"},
        )
        self.update_btn.on_click(self._refresh_info)

        # Auto-refresh process list on load
        # self._refresh_process_list()

        super().__init__(
            title="View Results",
            info="View and analyze calculation results.",
            widgets=[
                self.update_btn
                # self.refresh_button,
                # self.process_list_area,
                # ipw.HBox([self.process_selector]),
                # ipw.HBox([self.pk_input, self.load_button]),
            ],
            submittable=False,
            **kwargs,
        )

        self.render()
        # Observe calculation_pk changes to auto-load
        # self.model.observe(self._on_pk_change, names="calculation_pk")

    def render(self) -> None:
        """Render the wizard's uninitialised content."""
        if self.rendered:
            return
        # if self.model.blocked:
        #     msg = ipw.HTML(
        #         """
        #         <p>
        #             No process has been submitted...
        #         </p>
        #         """
        #     )
        #     self.children = [msg]
        # else:
        self.node_tree = ProcessNodesTreeWidget()
        # ipw.dlink((self.model, "process_uuid"), (self.node_tree, "value"))
        self.node_view = CustomAiidaNodeViewWidget()
        ipw.dlink(
            (self.node_tree, "selected_nodes"),
            (self.node_view, "node"),
            transform=lambda nodes: nodes[0] if nodes else None,
        )

        self.children = [
            self.info,
            self.node_tree,
            self.node_view,
            self.update_btn,
        ]
        self.rendered = True

    def _refresh_info(self, _) -> None:
        """Refresh the process information."""
        self.node_tree.update()

    # def _on_refresh_click(self, _) -> None:
    #     """Refresh the process list."""
    #     self._refresh_process_list()

    # def _refresh_process_list(self) -> None:
    #     """Load and display recent processes."""
    #     with self.process_list_area:
    #         self.process_list_area.clear_output()

    #         try:
    #             # Query recent calculations
    #             qb = QueryBuilder()
    #             qb.append(
    #                 ProcessNode,
    #                 filters={"attributes.process_label": {"in": ["Singlepoint", "GeomOpt", "MD"]}},
    #                 project=[
    #                     "id",
    #                     "ctime",
    #                     "attributes.process_label",
    #                     "attributes.process_state",
    #                     "attributes.exit_status",
    #                 ],
    #             )
    #             qb.order_by({ProcessNode: {"ctime": "desc"}})
    #             qb.limit(20)

    #             results = qb.all()

    #             if not results:
    #                 print("No MLIP calculations found. Submit one in Step 4!")
    #                 self.process_selector.options = []
    #                 return

    #             print("Recent Calculations:\n")
    #             print(f"{'PK':<6} {'Type':<15} {'State':<12} {'Exit':<6} {'Created'}")
    #             print("-" * 70)

    #             options = []
    #             for pk, ctime, label, state, exit_status in results:
    #                 status_icon = "OK" if exit_status == 0 else "ERR" if exit_status else "RUN"
    #                 time_str = ctime.strftime("%Y-%m-%d %H:%M")
    #                 print(f"{pk:<6} {label:<15} {state or 'N/A':<12} {status_icon:<6} {time_str}")

    #                 # Add to dropdown options
    #                 display = f"PK {pk} - {label} ({state or 'N/A'}) - {time_str}"
    #                 options.append((display, pk))

    #             self.process_selector.options = options
    #             if options:
    #                 self.process_selector.value = options[0][1]  # Select most recent

    #         except Exception as e:
    #             print(f"Error loading processes: {e}")

    #             traceback.print_exc()

    # def _on_process_select(self, change):
    #     """Handle process selection from dropdown."""
    #     if change["new"]:
    #         self.pk_input.value = change["new"]

    # def _on_pk_change(self, change):
    #     """Auto-load results when PK is set."""
    #     if change["new"] > 0:
    #         self.pk_input.value = change["new"]
    #         self._on_load_click(None)

    # def _on_load_click(self, button):
    #     """Load and display results for the given PK."""
    #     with self.logspace:
    #         self.logspace.clear_output()

    #         pk = self.pk_input.value
    #         if pk <= 0:
    #             print("Error: Please enter a valid process PK")
    #             self.status.failure("Invalid PK")
    #             return

    #         try:
    #             # Load the calculation node
    #             node = orm.load_node(pk)

    #             print(f"=== Calculation Results (PK: {pk}) ===\n")
    #             print(f"Type: {node.process_label}")
    #             print(f"State: {node.process_state}")
    #             print(f"Exit status: {node.exit_status}")
    #             print(f"Created: {node.ctime}")
    #             print(f"Finished: {node.mtime}\n")

    #             # Check if calculation finished successfully
    #             if node.exit_status != 0:
    #                 print(f"Warning: Calculation exited with status {node.exit_status}")
    #                 self.status.failure(f"Exit status: {node.exit_status}")
    #                 return

    #             # Get outputs
    #             if "results_dict" not in node.outputs:
    #                 print("Error: No results_dict output found")
    #                 self.status.failure("No results available")
    #                 return

    #             results = node.outputs.results_dict.get_dict()

    #             # Display energy
    #             energy_key = None
    #             for key in results.get("info", {}).keys():
    #                 if "energy" in key.lower():
    #                     energy_key = key
    #                     break

    #             if energy_key:
    #                 energy = results["info"][energy_key]
    #                 print(f"Energy: {energy:.6f} eV")

    #             # Display forces info
    #             force_key = None
    #             for key in results.keys():
    #                 if "force" in key.lower():
    #                     force_key = key
    #                     break

    #             if force_key:
    #                 forces = results[force_key]

    #                 force_magnitudes = np.linalg.norm(forces, axis=1)
    #                 print(f"Forces:")
    #                 print(f"   Max: {force_magnitudes.max():.4f} eV/Å")
    #                 print(f"   Mean: {force_magnitudes.mean():.4f} eV/Å")
    #                 print(f"   RMS: {np.sqrt((force_magnitudes**2).mean()):.4f} eV/Å\n")

    #             # Display stress if available
    #             stress_key = None
    #             for key in results.get("info", {}).keys():
    #                 if "stress" in key.lower():
    #                     stress_key = key
    #                     break

    #             if stress_key:
    #                 stress = results["info"][stress_key]
    #                 print(f"Stress tensor: {stress}\n")

    #             # Display structure info
    #             n_atoms = len(results["positions"])
    #             elements = set(results["numbers"])
    #             print(f"Structure:")
    #             print(f"   Atoms: {n_atoms}")
    #             print(f"   Elements: {', '.join(map(str, sorted(elements)))}")
    #             print(f"   PBC: {results.get('pbc', 'N/A')}\n")

    #             # Try to visualize structure with nglview
    #             try:

    #                 atoms = Atoms(
    #                     numbers=results["numbers"],
    #                     positions=results["positions"],
    #                     cell=results.get("cell"),
    #                     pbc=results.get("pbc", [True, True, True]),
    #                 )

    #                 print("Structure Visualization:")
    #                 view = nglview.show_ase(atoms)
    #                 view.add_unitcell()
    #                 view.add_ball_and_stick()
    #                 display(view)

    #             except Exception as e:
    #                 print(f"Warning: Could not display structure: {e}")

    #             self.status.value = f"<p style='color: green;'>Loaded results for PK {pk}</p>"

    #         except Exception as e:
    #             print(f"Error loading results: {e}")
    #             traceback.print_exc()
    #             self.status.value = f"<p style='color: red;'>Error: Failed to load results</p>"
