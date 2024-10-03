
from fastapi import APIRouter
from routes import langgpt

api_router = APIRouter()

api_router.include_router(
    langgpt.router, 
    prefix="", 
    tags=["langgpt"]
)