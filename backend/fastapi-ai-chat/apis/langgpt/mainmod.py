
from __future__ import annotations

from fastapi import HTTPException
from utilities.log_controler import LogControler
log_controler = LogControler()

from apis.langgpt.submod import ask_langchan_model_gpt

def ai_langchain_gpt_ask(data):
    result = None
    try:
        if data is None:
            raise HTTPException(status_code=400, detail='data is required.')
        
        result = ask_langchan_model_gpt(data)
        if "error_server" in result or "error_code" in result:
            raise HTTPException(status_code=400, detail='{}'.format(result.get('msg')))
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail='{}'.format(result.get('msg')))

        return result
    except Exception as e:
        log_controler.log_error(str(e), 'ai_langchain_gpt_ask')
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail='internal server error: {0}'.format(e))