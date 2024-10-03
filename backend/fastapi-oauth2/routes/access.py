from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from apis.access.mainmod import get_token
from core.auth import valid_access_token

router = APIRouter()

# Route to get access token
@router.post("/v1/token/")
async def generate_access_token(
    client_id: str = Header(...),
    client_secret: str = Header(...),
):
    return get_token(client_id, client_secret)

# Protected route that requires authentication
@router.get("/v1/protected/")
async def protected_authen(
    token: str = Depends(valid_access_token)
):
    return token