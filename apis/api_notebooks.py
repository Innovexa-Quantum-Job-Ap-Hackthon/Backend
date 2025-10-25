from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from notebooks.notebook_service import NotebookService
from notebooks.code_execution_service import CodeExecutionService
from notebooks.collaboration_server import collaboration_server
from Database.DBConfiguration.database import get_db
from Authorization.auth import get_current_user
from pydantic import BaseModel
import Database.DBmodels.database_models as models

router = APIRouter(prefix="/api/notebooks", tags=["notebooks"])

# Pydantic models for request/response validation

class NotebookCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    team_id: int

class NotebookUpdateRequest(BaseModel):
    name: str
    description: Optional[str] = None

class CellCreateRequest(BaseModel):
    notebook_id: int
    cell_type: str
    content: str
    order: int

class CellUpdateRequest(BaseModel):
    content: str

class VersionCreateRequest(BaseModel):
    commit_message: str

class ExecuteCodeRequest(BaseModel):
    code: str

class NotebookResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    team_id: int
    created_by_id: int
    created_at: str  # Use str for datetime serialization

@router.post("/", response_model=NotebookResponse)
def create_notebook(
    request: NotebookCreateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import logging
    logger = logging.getLogger("api_notebooks")
    logger.info(f"Create notebook request: {request}")
    service = NotebookService(db)
    try:
        notebook = service.create_notebook(request.name, request.description, request.team_id, current_user.id)
        # Broadcast new notebook creation to all team members (non-blocking)
        try:
            import asyncio
            asyncio.create_task(collaboration_server.broadcast_to_team(
                request.team_id,
                {
                    "type": "notebook_created",
                    "notebook_id": notebook.id,
                    "name": notebook.name,
                    "description": notebook.description,
                    "team_id": notebook.team_id
                },
                exclude_user_id=current_user.id
            ))
        except Exception as broadcast_error:
            logger.warning(f"Broadcast failed for notebook creation: {broadcast_error}")
            # Continue with notebook creation even if broadcast fails
        # Make notebook available to all team members by adding to their accessible notebooks
        # This is handled by the team membership and access control in NotebookService
        response_data = {
            "id": notebook.id,
            "name": notebook.name,
            "description": notebook.description,
            "team_id": notebook.team_id,
            "created_by_id": notebook.created_by_id,
            "created_at": notebook.created_at.isoformat()
        }
        logger.info(f"Create notebook response data: {response_data}")
        return NotebookResponse(**response_data)
    except Exception as e:
        logger.error(f"Error in create_notebook: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/team/{team_id}", response_model=list)
def get_team_notebooks(
    team_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        notebooks = service.get_team_notebooks(team_id, current_user.id)
        return [
            {
                "id": notebook.id,
                "name": notebook.name,
                "description": notebook.description,
                "team_id": notebook.team_id,
                "created_by_id": notebook.created_by_id,
                "created_at": notebook.created_at
            } for notebook in notebooks
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{notebook_id}", response_model=dict)
def get_notebook(
    notebook_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    notebook = service.get_notebook(notebook_id, current_user.id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    # Return notebook metadata and cells
    cells = service.get_notebook_cells(notebook_id, current_user.id)
    return {
        "id": notebook.id,
        "name": notebook.name,
        "description": notebook.description,
        "team_id": notebook.team_id,
        "cells": [
            {
                "id": cell.id,
                "cell_type": cell.cell_type,
                "content": cell.content,
                "order": cell.order
            } for cell in cells
        ]
    }

@router.put("/{notebook_id}", response_model=dict)
def update_notebook(
    notebook_id: int,
    request: NotebookUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        notebook = service.update_notebook(notebook_id, request.name, request.description, current_user.id)
        return {"message": "Notebook updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{notebook_id}", response_model=dict)
def delete_notebook(
    notebook_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        service.delete_notebook(notebook_id, current_user.id)
        return {"message": "Notebook deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{notebook_id}/cells", response_model=dict)
def create_cell(
    notebook_id: int,
    request: CellCreateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        cell = service.create_cell(notebook_id, request.cell_type, request.content, request.order, current_user.id)
        return {"cell_id": cell.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/cells/{cell_id}", response_model=dict)
def update_cell(
    cell_id: int,
    request: CellUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        cell = service.update_cell(cell_id, request.content, current_user.id)
        return {"message": "Cell updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/cells/{cell_id}", response_model=dict)
def delete_cell(
    cell_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        service.delete_cell(cell_id, current_user.id)
        return {"message": "Cell deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{notebook_id}/reorder_cells", response_model=dict)
def reorder_cells(
    notebook_id: int,
    cell_orders: List[dict],
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        service.reorder_cells(notebook_id, cell_orders, current_user.id)
        return {"message": "Cells reordered"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{notebook_id}/versions", response_model=dict)
def create_version(
    notebook_id: int,
    request: VersionCreateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        version = service.create_version(notebook_id, request.commit_message, current_user.id)
        return {"version_id": version.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{notebook_id}/versions", response_model=list)
def get_versions(
    notebook_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    versions = service.get_notebook_versions(notebook_id, current_user.id)
    return [
        {
            "id": v.id,
            "version_number": v.version_number,
            "commit_message": v.commit_message,
            "created_by_id": v.created_by_id,
            "created_at": v.created_at
        } for v in versions
    ]

@router.post("/{notebook_id}/restore_version/{version_id}", response_model=dict)
def restore_version(
    notebook_id: int,
    version_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        notebook = service.restore_version(notebook_id, version_id, current_user.id)
        return {"message": "Notebook restored to version"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{notebook_id}/fork", response_model=dict)
def fork_notebook(
    notebook_id: int,
    name: str,
    description: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = NotebookService(db)
    try:
        notebook = service.fork_notebook(notebook_id, name, description, current_user.id)
        return {"notebook_id": notebook.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates", response_model=list)
def get_templates():
    service = NotebookService(None)
    templates = service.get_notebook_templates()
    return templates

@router.post("/{notebook_id}/execute", response_model=dict)
def execute_code(
    notebook_id: int,
    request: ExecuteCodeRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify user has access to the notebook
    service = NotebookService(db)
    notebook = service.get_notebook(notebook_id, current_user.id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    executor = CodeExecutionService()
    validation = executor.validate_code(request.code)
    if not validation["valid"]:
        return {"success": False, "output": "", "error": " ; ".join(validation["issues"]), "plots": [], "circuit_diagrams": []}
    result = executor.execute_code(request.code)
    return result
