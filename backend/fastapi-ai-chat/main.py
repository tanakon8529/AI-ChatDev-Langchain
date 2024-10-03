

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from settings.configs import API_VERSION, API_PATH_FASTAPI_AI_CHAT, API_DOC
from endpoint import api_router

from utilities.log_controler import LogControler
log_controler = LogControler(port=API_PATH_FASTAPI_AI_CHAT)

app = FastAPI(
    title="FastAPI Ai-Chat",
    description="FastAPI Ai-Chat",
    version=API_VERSION,
    docs_url=f"{API_PATH_FASTAPI_AI_CHAT}{API_DOC}",
    redoc_url=None,
    openapi_url=f"{API_PATH_FASTAPI_AI_CHAT}{API_DOC}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=API_PATH_FASTAPI_AI_CHAT)