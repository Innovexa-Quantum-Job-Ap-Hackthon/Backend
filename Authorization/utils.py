# utils.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

# -------------------------------
# SECURITY SETTINGS
# -------------------------------
# NEVER hardcode secrets in production. Use environment variables.
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30000000))

# -------------------------------
# PASSWORD HASHING CONTEXT
# -------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify that the plain password matches the hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash the given plain password.
    """
    return pwd_context.hash(password)

# -------------------------------
# JWT TOKEN FUNCTIONS
# -------------------------------
def create_access_token(data: dict) -> str:
    """
    Create a JWT access token with an expiration time.
    The 'sub' claim is typically used for the user's identifier (email or id).
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """
    Decode a JWT token and return its payload.
    Raises JWTError if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise e
