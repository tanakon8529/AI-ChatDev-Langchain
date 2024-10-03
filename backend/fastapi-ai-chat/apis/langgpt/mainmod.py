
from __future__ import annotations

from fastapi import HTTPException
from utilities.log_controler import LogControler
log_controler = LogControler()

from apis.langgpt.submod import test

def ai_langchain_gpt_ask():
    result = None
    try:
        data = test()
        result = {
            "msg": "success",
            "data": data
        }

        if result == None:
            raise HTTPException(status_code=404, detail='Not Found')
        if "error_code" in result:
            raise HTTPException(status_code=400, detail='{}'.format(result["msg"]))
        if "error_server" in result:
            raise HTTPException(status_code=500, detail='{}'.format(result["msg"]))

        return result
    except Exception as e:
        log_controler.log_error(str(e), 'user_signin')
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail='internal server error: {0}'.format(e))