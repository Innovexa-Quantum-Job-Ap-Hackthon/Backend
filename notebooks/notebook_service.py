import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

# Import database models
from Database.DBmodels.notebook_models import Notebook, NotebookCell, NotebookVersion
from Database.DBmodels.database_models import User, Team, TeamMember

logger = logging.getLogger(__name__)

class NotebookService:
    def __init__(self, db: Session):
        self.db = db

    def create_notebook(self, name: str, description: Optional[str], team_id: int, user_id: int) -> Notebook:
        """Create a new notebook for a team"""
        # Verify team exists
        team = self.db.query(Team).filter(Team.id == team_id).first()
        if not team:
            raise ValueError("Team not found")

        # Verify user is team member
        membership = self.db.query(TeamMember).filter(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id
        ).first()
        if not membership:
            raise ValueError("User is not a member of this team")

        # Create notebook
        db_notebook = Notebook(
            name=name,
            description=description,
            team_id=team_id,
            created_by_id=user_id
        )
        self.db.add(db_notebook)
        self.db.flush()

        # Create initial version
        initial_snapshot = {
            "name": name,
            "description": description,
            "cells": []
        }
        initial_version = NotebookVersion(
            notebook_id=db_notebook.id,
            version_number=1,
            commit_message="Initial notebook creation",
            created_by_id=user_id,
            snapshot=initial_snapshot
        )
        self.db.add(initial_version)
        self.db.commit()
        self.db.refresh(db_notebook)

        # Save as .ipynb file
        notebooks_dir = os.path.join(os.getcwd(), 'notebooks')
        os.makedirs(notebooks_dir, exist_ok=True)

        # Sanitize name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.ipynb"
        filepath = os.path.join(notebooks_dir, filename)

        # Handle duplicates
        counter = 1
        while os.path.exists(filepath):
            filename = f"{safe_name}_{counter}.ipynb"
            filepath = os.path.join(notebooks_dir, filename)
            counter += 1

        # Create .ipynb content
        ipynb_content = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ipynb_content, f, indent=2, ensure_ascii=False)

        logger.info(f"Notebook '{name}' created by user {user_id} in team {team_id}, saved as {filename}")
        return db_notebook

    def get_notebook(self, notebook_id: int, user_id: int) -> Optional[Notebook]:
        """Get a notebook with access control"""
        notebook = self.db.query(Notebook).filter(Notebook.id == notebook_id).first()
        if not notebook:
            return None

        # Check if user is team member
        membership = self.db.query(TeamMember).filter(
            TeamMember.team_id == notebook.team_id,
            TeamMember.user_id == user_id
        ).first()
        if not membership:
            raise ValueError("Access denied: User is not a member of this team")

        return notebook

    def get_team_notebooks(self, team_id: int, user_id: int) -> List[Notebook]:
        """Get all notebooks for a team"""
        # Verify user is team member
        membership = self.db.query(TeamMember).filter(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id
        ).first()
        if not membership:
            raise ValueError("User is not a member of this team")

        return self.db.query(Notebook).filter(Notebook.team_id == team_id).all()

    def update_notebook(self, notebook_id: int, name: str, description: str, user_id: int) -> Notebook:
        """Update notebook metadata"""
        notebook = self.get_notebook(notebook_id, user_id)
        if not notebook:
            raise ValueError("Notebook not found")

        notebook.name = name
        notebook.description = description
        self.db.commit()
        self.db.refresh(notebook)

        logger.info(f"Notebook {notebook_id} updated by user {user_id}")
        return notebook

    def delete_notebook(self, notebook_id: int, user_id: int) -> bool:
        """Delete a notebook"""
        notebook = self.get_notebook(notebook_id, user_id)
        if not notebook:
            raise ValueError("Notebook not found")

        # Check if user is the creator
        if notebook.created_by_id != user_id:
            raise ValueError("Only notebook creators can delete notebooks")

        self.db.delete(notebook)
        self.db.commit()

        logger.info(f"Notebook {notebook_id} deleted by user {user_id}")
        return True

    def create_cell(self, notebook_id: int, cell_type: str, content: str, order: int, user_id: int) -> NotebookCell:
        """Create a new cell in a notebook"""
        notebook = self.get_notebook(notebook_id, user_id)
        if not notebook:
            raise ValueError("Notebook not found")

        cell = NotebookCell(
            notebook_id=notebook_id,
            cell_type=cell_type,
            content=content,
            order=order
        )
        self.db.add(cell)
        self.db.commit()
        self.db.refresh(cell)

        logger.info(f"Cell created in notebook {notebook_id} by user {user_id}")
        return cell

    def update_cell(self, cell_id: int, content: str, user_id: int) -> NotebookCell:
        """Update cell content"""
        cell = self.db.query(NotebookCell).filter(NotebookCell.id == cell_id).first()
        if not cell:
            raise ValueError("Cell not found")

        # Check notebook access
        self.get_notebook(cell.notebook_id, user_id)

        cell.content = content
        self.db.commit()
        self.db.refresh(cell)

        logger.info(f"Cell {cell_id} updated by user {user_id}")
        return cell

    def delete_cell(self, cell_id: int, user_id: int) -> bool:
        """Delete a cell"""
        cell = self.db.query(NotebookCell).filter(NotebookCell.id == cell_id).first()
        if not cell:
            raise ValueError("Cell not found")

        # Check notebook access
        self.get_notebook(cell.notebook_id, user_id)

        self.db.delete(cell)
        self.db.commit()

        logger.info(f"Cell {cell_id} deleted by user {user_id}")
        return True

    def get_notebook_cells(self, notebook_id: int, user_id: int) -> List[NotebookCell]:
        """Get all cells for a notebook"""
        self.get_notebook(notebook_id, user_id)  # Access check

        return self.db.query(NotebookCell).filter(
            NotebookCell.notebook_id == notebook_id
        ).order_by(NotebookCell.order).all()

    def reorder_cells(self, notebook_id: int, cell_orders: List[Dict[str, int]], user_id: int) -> bool:
        """Reorder cells in a notebook"""
        self.get_notebook(notebook_id, user_id)  # Access check

        for cell_order in cell_orders:
            cell_id = cell_order.get("cell_id")
            order = cell_order.get("order")

            cell = self.db.query(NotebookCell).filter(
                NotebookCell.id == cell_id,
                NotebookCell.notebook_id == notebook_id
            ).first()

            if cell:
                cell.order = order

        self.db.commit()
        logger.info(f"Cells reordered in notebook {notebook_id} by user {user_id}")
        return True

    def create_version(self, notebook_id: int, commit_message: str, user_id: int) -> NotebookVersion:
        """Create a new version/commit of the notebook"""
        notebook = self.get_notebook(notebook_id, user_id)
        if not notebook:
            raise ValueError("Notebook not found")

        # Get current version number
        latest_version = self.db.query(NotebookVersion).filter(
            NotebookVersion.notebook_id == notebook_id
        ).order_by(desc(NotebookVersion.version_number)).first()

        version_number = (latest_version.version_number + 1) if latest_version else 1

        # Create snapshot of current notebook state
        cells = self.get_notebook_cells(notebook_id, user_id)
        snapshot = {
            "name": notebook.name,
            "description": notebook.description,
            "cells": [
                {
                    "id": cell.id,
                    "type": cell.cell_type,
                    "content": cell.content,
                    "order": cell.order
                } for cell in cells
            ]
        }

        version = NotebookVersion(
            notebook_id=notebook_id,
            version_number=version_number,
            commit_message=commit_message,
            created_by_id=user_id,
            snapshot=snapshot
        )

        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)

        logger.info(f"Version {version_number} created for notebook {notebook_id} by user {user_id}")
        return version

    def get_notebook_versions(self, notebook_id: int, user_id: int) -> List[NotebookVersion]:
        """Get version history for a notebook"""
        self.get_notebook(notebook_id, user_id)  # Access check

        return self.db.query(NotebookVersion).filter(
            NotebookVersion.notebook_id == notebook_id
        ).order_by(desc(NotebookVersion.created_at)).all()

    def restore_version(self, notebook_id: int, version_id: int, user_id: int) -> Notebook:
        """Restore notebook to a previous version"""
        notebook = self.get_notebook(notebook_id, user_id)
        if not notebook:
            raise ValueError("Notebook not found")

        version = self.db.query(NotebookVersion).filter(
            NotebookVersion.id == version_id,
            NotebookVersion.notebook_id == notebook_id
        ).first()

        if not version:
            raise ValueError("Version not found")

        # Restore notebook metadata
        snapshot = version.snapshot
        notebook.name = snapshot.get("name", notebook.name)
        notebook.description = snapshot.get("description", notebook.description)

        # Clear existing cells
        self.db.query(NotebookCell).filter(NotebookCell.notebook_id == notebook_id).delete()

        # Restore cells from snapshot
        for cell_data in snapshot.get("cells", []):
            cell = NotebookCell(
                notebook_id=notebook_id,
                cell_type=cell_data.get("type", "code"),
                content=cell_data.get("content", ""),
                order=cell_data.get("order", 0)
            )
            self.db.add(cell)

        self.db.commit()
        self.db.refresh(notebook)

        logger.info(f"Notebook {notebook_id} restored to version {version.version_number} by user {user_id}")
        return notebook

    def fork_notebook(self, notebook_id: int, name: str, description: str, user_id: int) -> Notebook:
        """Create a fork/copy of a notebook"""
        original_notebook = self.get_notebook(notebook_id, user_id)
        if not original_notebook:
            raise ValueError("Notebook not found")

        # Create new notebook
        forked_notebook = self.create_notebook(name, description, original_notebook.team_id, user_id)

        # Copy cells from original
        original_cells = self.get_notebook_cells(notebook_id, user_id)
        for cell in original_cells:
            new_cell = NotebookCell(
                notebook_id=forked_notebook.id,
                cell_type=cell.cell_type,
                content=cell.content,
                order=cell.order
            )
            self.db.add(new_cell)

        self.db.commit()

        logger.info(f"Notebook {notebook_id} forked as {forked_notebook.id} by user {user_id}")
        return forked_notebook

    def get_notebook_templates(self) -> List[Dict[str, Any]]:
        """Get available notebook templates"""
        templates = [
            {
                "id": "qft",
                "name": "Quantum Fourier Transform",
                "description": "Implementation of the Quantum Fourier Transform algorithm",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "# Quantum Fourier Transform\n\nThis notebook demonstrates the Quantum Fourier Transform algorithm."
                    },
                    {
                        "type": "code",
                        "content": "from qiskit import QuantumCircuit, transpile\nfrom qiskit.providers.basic_provider import BasicProvider\nfrom qiskit.visualization import circuit_drawer\nimport matplotlib.pyplot as plt\n\n# Create a quantum circuit with 3 qubits\nqc = QuantumCircuit(3)\n\n# Apply QFT\nqc.h(0)\nqc.cp(3.14159/2, 1, 0)\nqc.cp(3.14159/4, 2, 0)\nqc.h(1)\nqc.cp(3.14159/2, 2, 1)\nqc.h(2)\n\n# Swap qubits\nqc.swap(0, 2)\n\nprint('QFT Circuit:')\nprint(qc)"
                    }
                ]
            },
            {
                "id": "grover",
                "name": "Grover's Algorithm",
                "description": "Implementation of Grover's search algorithm",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "# Grover's Algorithm\n\nThis notebook demonstrates Grover's quantum search algorithm."
                    },
                    {
                        "type": "code",
                        "content": "from qiskit import QuantumCircuit, transpile\nfrom qiskit.providers.basic_provider import BasicProvider\nfrom qiskit.visualization import circuit_drawer\nimport matplotlib.pyplot as plt\n\n# Create a quantum circuit with 2 qubits (for 4 possible states)\nqc = QuantumCircuit(2, 2)\n\n# Initialize superposition\nqc.h([0, 1])\n\n# Oracle for state |11>\nqc.cz(0, 1)\n\n# Diffusion operator\nqc.h([0, 1])\nqc.z([0, 1])\nqc.cz(0, 1)\nqc.h([0, 1])\n\n# Measure\nqc.measure([0, 1], [0, 1])\n\nprint('Grover Circuit:')\nprint(qc)"
                    }
                ]
            },
            {
                "id": "teleportation",
                "name": "Quantum Teleportation",
                "description": "Implementation of quantum teleportation protocol",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "# Quantum Teleportation\n\nThis notebook demonstrates quantum teleportation."
                    },
                    {
                        "type": "code",
                        "content": "from qiskit import QuantumCircuit, transpile\nfrom qiskit.providers.basic_provider import BasicProvider\nfrom qiskit.visualization import circuit_drawer\nimport matplotlib.pyplot as plt\n\n# Create quantum circuit with 3 qubits and 2 classical bits\nqc = QuantumCircuit(3, 2)\n\n# Prepare the state to be teleported (qubit 0)\nqc.h(0)  # Create superposition\n\n# Create entanglement between qubits 1 and 2\nqc.h(1)\nqc.cx(1, 2)\n\n# Teleportation protocol\nqc.cx(0, 1)\nqc.h(0)\nqc.measure(0, 0)\nqc.measure(1, 1)\n\n# Apply corrections based on measurement results\nqc.x(2).c_if(1, 1)\nqc.z(2).c_if(0, 1)\n\nprint('Quantum Teleportation Circuit:')\nprint(qc)"
                    }
                ]
            }
        ]

        return templates

    def create_notebook_from_template(self, template_id: str, team_id: int, user_id: int) -> Notebook:
        """Create a notebook from a template"""
        templates = self.get_notebook_templates()
        template = next((t for t in templates if t["id"] == template_id), None)
        if not template:
            raise ValueError("Template not found")

        # Create notebook
        notebook = self.create_notebook(
            name=template["name"],
            description=template["description"],
            team_id=team_id,
            user_id=user_id
        )

        # Add template cells
        for i, cell_data in enumerate(template["cells"]):
            cell = NotebookCell(
                notebook_id=notebook.id,
                cell_type=cell_data["type"],
                content=cell_data["content"],
                order=i
            )
            self.db.add(cell)

        self.db.commit()

        logger.info(f"Notebook created from template '{template_id}' by user {user_id}")
        return notebook
