from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from Database.DBConfiguration.database import get_db

import Database.DBmodels.database_models as  models, Database.DBmodels.schemas as schemas, Authorization.utils as utils

from Authorization.utils import SECRET_KEY, ALGORITHM
 
# OAuth2 scheme for token verification
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency: Get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

router = APIRouter(tags=['Authentication'])

@router.post("/signup", response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Creates a new user with hashed password.
    Handles duplicate emails and unexpected errors.
    """
    try:
        # Check if email already exists
        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash the password
        hashed_password = utils.get_password_hash(user.password)

        # Create user object
        db_user = models.User(
            email=user.email,
            hashed_password=hashed_password,
            name=user.name
        )

        # Add to DB
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        return db_user

    except HTTPException:
        # Raise HTTPExceptions as they are
        raise
    except Exception as e:
        # Catch all other unexpected errors
        print(f"Signup Error: {e}")  # Log the error to terminal
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during signup"
        )

@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = utils.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/verify")
def verify_token(current_user: models.User = Depends(get_current_user)):
    return {"user": {"id": current_user.id, "email": current_user.email, "name": current_user.name}}

@router.get("/user/ibm-api-key")
def get_ibm_api_key(current_user: models.User = Depends(get_current_user)):
    import logging
    logging.info(f"Fetching IBM keys for user: {current_user.email}")
    logging.info(f"IBM API Key present: {'Yes' if current_user.ibm_api_key_encrypted else 'No'}")
    logging.info(f"IBM Instance Key present: {'Yes' if current_user.ibm_instance_key_encrypted else 'No'}")
    return {
        "ibm_api_key": current_user.ibm_api_key_encrypted,
        "ibm_instance_key": current_user.ibm_instance_key_encrypted
    }


# @router.post("/api/job/{job_id}")
# def post_job_logs(job_id: str,db: Session = Depends(get_db)):
#     data=get_job_status(job_id)
#     '''
#     # db_job_logs = models.JobLogs(**JobLogs.dict())
#     # print(db_job_logs)
#     # db.add(db_job_logs)
#     # db.commit()
#     # db.refresh(db_job_logs)
#     # return db_job_logs
#     '''
#     return data







from fastapi import Body

class IbmKeysSchema(schemas.BaseModel):
    ibm_api_key: str
    ibm_instance_key: str

@router.post("/user/ibm-api-key")
def set_ibm_api_key(keys: IbmKeysSchema, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Validate IBM credentials before saving
    try:
        from main import get_ibm_service_for_user
        # Create a temporary user object with the new keys to test them
        temp_user = type('TempUser', (), {
            'id': current_user.id,  # Add the missing id attribute
            'ibm_api_key_encrypted': keys.ibm_api_key,
            'ibm_instance_key_encrypted': keys.ibm_instance_key,
            'email': current_user.email
        })()
        # This will raise an HTTPException if credentials are invalid
        get_ibm_service_for_user(temp_user)
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "Invalid API key" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Invalid IBM API key or instance key. Please check your credentials and try again."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to validate IBM credentials: {str(e)}"
            )

    # Save the validated keys
    current_user.ibm_api_key_encrypted = keys.ibm_api_key
    current_user.ibm_instance_key_encrypted = keys.ibm_instance_key
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return {"message": "IBM API and Instance keys validated and saved successfully."}
