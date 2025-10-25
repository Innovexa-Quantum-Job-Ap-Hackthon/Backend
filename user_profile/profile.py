from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import Database.DBmodels.database_models as models
from Authorization.auth import get_current_user
from Database.DBConfiguration.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

@router.get("/api/profile")
async def get_profile(current_user: models.User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "name": current_user.name,
        "apiKey": current_user.ibm_api_key_encrypted or ""
    }

@router.put("/api/profile/api-key")
async def update_api_key(api_key: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.ibm_api_key_encrypted = api_key
    db.commit()
    return {"message": "API Key updated"}
