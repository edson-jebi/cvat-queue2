"""Authentication routes and utilities."""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import jwt, JWTError

from app.database import get_db
from app.models import User
from app.services.cvat_client import CVATClient

router = APIRouter()

SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def create_token(username: str, cvat_host: str) -> str:
    """Create JWT token for session."""
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    data = {"sub": username, "host": cvat_host, "exp": expire}
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> Optional[User]:
    """Get current user from session cookie."""
    token = request.cookies.get("session")
    if not token:
        return None

    data = decode_token(token)
    if not data:
        return None

    result = await db.execute(select(User).where(User.username == data["sub"]))
    return result.scalar_one_or_none()


async def require_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Require authenticated user or redirect to login."""
    user = await get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def require_admin(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Require admin user."""
    user = await require_user(request, db)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    cvat_host: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Login with CVAT credentials."""
    # Ensure host has protocol
    if not cvat_host.startswith("http"):
        cvat_host = f"https://{cvat_host}"

    # Try to authenticate with CVAT
    client = CVATClient(cvat_host)
    success, result = await client.login(username, password)
    await client.close()

    if not success:
        raise HTTPException(status_code=401, detail=result)

    # Create or update local user
    db_user = await db.execute(select(User).where(User.username == username))
    user = db_user.scalar_one_or_none()

    if user:
        user.cvat_token = result
        user.cvat_host = cvat_host
    else:
        user = User(
            username=username,
            cvat_token=result,
            cvat_host=cvat_host,
            is_admin=False
        )
        db.add(user)

    await db.commit()

    # Set session cookie
    token = create_token(username, cvat_host)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        max_age=TOKEN_EXPIRE_HOURS * 3600
    )
    return response


@router.get("/logout")
async def logout():
    """Clear session and redirect to login."""
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session")
    return response
