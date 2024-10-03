

from __future__ import annotations

from fastapi import HTTPException

from core.auth import authenticate_user, store_access_token
from utilities.log_controler import LogControler
log_controler = LogControler()

def get_token(client_id, client_secret):

    try:
        access_token = authenticate_user(client_id, client_secret)
        if not access_token:
            raise HTTPException(status_code=401, detail='Access Denied')
        
        result = store_access_token(access_token)
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail='{}'.format(result))

        return result
    except Exception as e:
        log_controler.log_error(str(e), 'get_token')
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail='internal server error: {0}'.format(e))