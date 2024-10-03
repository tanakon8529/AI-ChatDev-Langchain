
import os
from dotenv import load_dotenv

load_dotenv("./.env")

#### MICROSERVICE ####
MICROSERVICE_NAME_FASTAPI_OAUTH2 = "fastapi-oauth2"
MICROSERVICE_NAME_FASTAPI_AI_CHAT = "fastapi-ai-chat"

def set_microservice_name_by_api_path(api_path):
    if api_path == API_PATH_FASTAPI_OAUTH2:
        return MICROSERVICE_NAME_FASTAPI_OAUTH2
    elif api_path == API_PATH_FASTAPI_AI_CHAT:
        return MICROSERVICE_NAME_FASTAPI_AI_CHAT
    else:
        return "Unknown"
    
API_VERSION = os.environ["API_VERSION"]
API_PATH_FASTAPI_OAUTH2 = os.environ["API_PATH_FASTAPI_OAUTH2"]
API_PATH_FASTAPI_AI_CHAT = os.environ["API_PATH_FASTAPI_AI_CHAT"]
API_DOC = os.environ["API_DOC"]
HOST = os.environ["HOST"]

PORT_FASTAPI_OAUTH2 = os.environ["PORT_FASTAPI_OAUTH2"]
PORT_FASTAPI_USER = os.environ["PORT_FASTAPI_AI_CHAT"]

#### AUTHENTICATION ####
USERNAME_ADMIN = os.environ["USERNAME_ADMIN"]
PASSWORD_ADMIN = os.environ["PASSWORD_ADMIN"]
SECRET_KEY = os.environ["SECRET_KEY"] + "="

#### REDIS ####
HOST_REDIS = os.environ["HOST_REDIS"]
PORT_REDIS = os.environ["PORT_REDIS"]
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

#### OpenAI ####
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_ID = os.environ["MODEL_ID"]
PERSIST_DIRECTORY = os.environ["PERSIST_DIRECTORY"]
PDF_PATH = os.environ["PDF_PATH"]