from __future__ import annotations

from fastapi import HTTPException
from utilities.log_controler import LogControler
log_controler = LogControler()

from apis.langgpt.submod import ask_langchain_model_gpt, test_chatbot_faiss

async def ai_langchain_gpt_ask(data):
    result = None
    try:
        if data is None:
            raise HTTPException(status_code=400, detail='data is required.')
        
        result = await ask_langchain_model_gpt(data)
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
        
async def ai_langchain_gpt_test():
    result = None
    try:
        result = await test_chatbot_faiss()
        if not result:
            raise HTTPException(status_code=500, detail='Failed to run tests.')
        
        result = {
            "msg": "success",
            "data": {
                "result": result
            }
        }
        return result
    except Exception as e:
        log_controler.log_error(str(e), 'ai_langchain_gpt_test')
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail='internal server error: {0}'.format(e))
        