from __future__ import annotations

from typing import Dict, Any
from fastapi import HTTPException

from core.models import DynamicBaseModel
from utilities.chatbot_faiss_test import ChatbotFAISSTest
from utilities.chatbot_faiss import ChatbotFAISS
from utilities.log_controler import LogControler

log_controller = LogControler()
chat_bot = ChatbotFAISS()

async def ask_langchain_model_gpt(data: DynamicBaseModel) -> Dict[str, Any]:
    question = data.question.strip() if data.question else None
    if not question:
        log_controller.log_error("Received empty question.", 'ask_langchain_model_gpt')
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    answer_response = await chat_bot.process_query(question)
    if "error_code" in answer_response:
        log_controller.log_error(f"Error from chat_bot: {answer_response.get('msg')}", 'ask_langchain_model_gpt')
        raise HTTPException(status_code=400, detail=answer_response.get('msg', 'Bad request.'))

    data_field = answer_response.get("data", {})
    answer = data_field.get("answer")
    type_res = data_field.get("type_res")

    if not answer:
        log_controller.log_error("No answer returned from chat_bot.", 'ask_langchain_model_gpt')
        raise HTTPException(status_code=500, detail="Failed to retrieve answer.")

    result = {
        "msg": "success",
        "data": {
            "answer": answer,
            "type_res": type_res
        }
    }
    return result

async def test_chatbot_faiss() -> None:
    try:
        tester = ChatbotFAISSTest()
        number_of_questions = 3 
        # Topic of generated questions, e.g. "AP Thailand, บริษัท เอพี (ไทยแลนด์) จำกัด"
        topic = "AP Thailand, บริษัท เอพี (ไทยแลนด์) จำกัด"
        await tester.run_tests(number_of_questions, topic)
        return True
    except Exception as e:
        return False

