"""Structure selection wizard step."""

import io
from pathlib import Path
from typing import Any

import ipywidgets as ipw
from aiida.orm import SinglefileData, StructureData
from aiidalab_widgets_base import SmilesWidget, WizardAppWidgetStep
from alc_aiidalab_widgets.layouts import Step
from alc_aiidalab_widgets.widgets.database import AiiDADatabaseQueryWidget
from alc_aiidalab_widgets.widgets.file_handling import FileUploadWidget
from alc_aiidalab_widgets.widgets.structure import StructureViewWidget
from ase.io import read as ase_read

from aiidalab_mlip.models.structure import StructureModel
from aiidalab_mlip.util import tab_from_dict
from aiidalab_mlip.wrappers import err_handler


class StructureWizardStep(Step, WizardAppWidgetStep):
    """Wizard step for structure selection."""

    def __init__(self, model: StructureModel, /, **kwargs: Any) -> None:
        """Initialize structure wizard step."""
        self.model = model

        # upload file
        self.file_uploader = FileUploadWidget(description="Structure file: ")
        self.file_input_widget = ipw.VBox([self.file_uploader])

        # AiiDA database
        self.database_widget = AiiDADatabaseQueryWidget(
            title="AiiDA Database",
            query=[SinglefileData, StructureData],
        )

        self.smiles_widget = SmilesWidget(title="SMILES")

        self.tabs = tab_from_dict(
            ipw.Tab,
            {
                "Upload File": self.file_input_widget,
                "AiiDA Database": self.database_widget,
                "SMILES String": self.smiles_widget,
            },
        )

        self.viewer = StructureViewWidget()

        super().__init__(
            title="Select or Upload Structure",
            info="Choose a structure.",
            widgets=(
                self.tabs,
                self.viewer,
            ),
            **kwargs,
        )
        ipw.dlink((self.file_uploader, "file"), (self.model, "structure"))
        self.file_uploader.file_upload.observe(self._on_file_upload, "value")
        self.database_widget.observe(self._on_database_search, "data_object")
        self.smiles_widget.observe(self._on_smiles_generation, "structure")

    def _on_file_upload(self, change: dict) -> None:
        """When file upload button is pressed."""
        self.status.clear()

        if not change["new"]:
            return

        # Get uploaded file
        filename = self.file_uploader.filename()
        ext: str | None = Path(filename).suffix.removeprefix(".")
        content = self.file_uploader.get_file_contents()

        file = io.StringIO()
        file.write(content.read().decode("ascii"))
        file.seek(0)

        with err_handler(self, "reading file"):
            structure = ase_read(file, format=ext)

        if self.status.status is self.status._Stat.FAILURE:
            return

        self.viewer.assign_structure_from_ase(structure)

        # Store in model
        assert isinstance(self.model, StructureModel)

        self.model.structure = structure
        self.model.filename = filename

        # Update status
        self.status.success(f"Loaded {filename}: {len(structure)} atoms.")

        # Display structure info
        with self.logspace:
            self.logspace.clear_output()
            print(f"Formula: {structure.get_chemical_formula()}")
            print(f"Number of atoms: {len(structure)}")
            print(f"Cell: {structure.get_cell()}")

    def _on_smiles_generation(self, change: dict) -> None:
        """When SMILES string is inputted."""
        if change["new"] == change["old"]:
            return

        self.viewer.assign_structure_from_ase(change["new"])
        self.model.structure = change["new"]

    def _on_database_search(self, change: dict) -> None:
        """When data is loaded from AiiDA database."""
        if change["new"] == change["old"]:
            return

        if isinstance(change["new"], SinglefileData):
            self.model.filename = change["new"]
            self._on_file_upload(change)
        elif isinstance(change["new"], StructureData):
            self.model.structure = change["new"]._get_object_ase()
            self.viewer.assign_structure_from_ase(self.model.structure)
        else:
            self.viewer.assign_structure_from_ase(None)

    def submit(self, b) -> None:
        """Submit the structure step."""
        with self.logspace:
            if self.model.structure or self.model.filename:
                self.file_uploader.disable(True)
                self.database_widget.disable(True)
                self.model.submitted = True
                self.status.success("Structure submitted.")
            else:
                self.model.submitted = False
                self.status.success("No structure defined.")

            super().submit(b)
