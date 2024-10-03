

from fastapi import APIRouter
from routes import access

api_router = APIRouter()

api_router.include_router(
    access.router, 
    prefix="", 
    tags=["access"]
)