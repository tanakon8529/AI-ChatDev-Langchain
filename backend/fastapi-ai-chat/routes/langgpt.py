
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Header

from core.auth import valid_access_token
from core.models import DynamicBaseModel

from apis.langgpt.mainmod import ai_langchain_gpt_ask

router = APIRouter()

@router.post("/v1/ask/")
async def ask_ai_langchain_gpt(
    data: Optional[DynamicBaseModel] = None,
    _: str = Depends(valid_access_token),
):
    return ai_langchain_gpt_ask(data)
