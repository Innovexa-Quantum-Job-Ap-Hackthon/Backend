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
                "created_at": notebook.created_at,
                "updated_at": notebook.updated_at
            } for notebook in notebooks
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
