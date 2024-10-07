from __future__ import annotations

from utilities.chatbot_faiss import ChatbotFAISS
from utilities.log_controler import LogControler
log_controler = LogControler()
chat_bot = ChatbotFAISS()

def ask_langchan_model_gpt(data):
    # data is DynamicBaseModel by Pydantic
    # convert data to dict
    data_dict = data.dict()
    question = data_dict.get('question')
    if question is None:
        return {"error_code": "01", "msg": "Question is required."}

    answer_response = chat_bot.process_query(question)
    if "error_code" in answer_response:
        return answer_response

    result = {
        "msg": "success",
        "data": {
            "answer": answer_response.get("answer"),
            "type_res": answer_response.get("type_res")
        }
    }
    return result
