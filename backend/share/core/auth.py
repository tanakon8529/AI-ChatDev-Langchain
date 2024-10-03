
import json

from uuid import uuid4
from typing import Dict
from datetime import timedelta
from fastapi import Header, HTTPException

from utilities.redis_connector import get_client
from utilities.log_controler import LogControler
log_controler = LogControler()

from settings.configs import USERNAME_ADMIN, PASSWORD_ADMIN, SECRET_KEY
# Define token expiration time (in minutes)
token_expire_minutes = 60
server_redis = "queue"

username_pack = [USERNAME_ADMIN]
password_pack = [PASSWORD_ADMIN]
secretKey = SECRET_KEY

# for other /access/protected
def valid_access_token(token: str = Header(...)):
    if "Bearer" in token:
        token = token.replace("Bearer", "").replace(" ", "")
    else:
        raise HTTPException(status_code=401, detail='Invalid access token')

    # Check if access token is valid
    redis = get_client()
    token_key = f'token:{token}'
    if not redis.get(token_key):
        raise HTTPException(status_code=401, detail='Invalid access token')

    return {'detail': 'Valid access token!'}

# Authenticate user and return access token
def authenticate_user(username: str, password: str):
    # Replace this with your own authentication logic
    access_token = None
    if username in username_pack and password in password_pack:
        access_token = str(uuid4())

    return access_token

# Store access token in Redis and return token info
def store_access_token(access_token: str):
    try:
        # Generate token key
        token_key = f'token:{access_token}'
        # Store access token with expiration time
        redis = get_client()
        redis.setex(token_key, timedelta(minutes=token_expire_minutes), access_token)
        # Return token info
        result = {'access_token': access_token, 'token_type': 'Bearer'}
    except Exception as e:
        log_controler.log_error(str(e), 'store_access_token')
        result = e

    return result