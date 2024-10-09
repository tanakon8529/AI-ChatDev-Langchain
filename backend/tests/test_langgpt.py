from settings.configs import HOST, PORT_FASTAPI_AI_CHAT
from utilitys.http_request import make_request, log_result

# Assuming these are set up as environment variables or constants
if HOST == "api.apthai.com":
    BASE_URL = f"http://localhost:{PORT_FASTAPI_AI_CHAT}/langgpt"
else:
    BASE_URL = f"http://{HOST}:{PORT_FASTAPI_AI_CHAT}/langgpt"

def test_post_ask():
    """Test for asking a question."""
    data = {
        "question": "สรุปการประเมินศักยภาพคู่ค้าล่าสุดให้หน่อย"
    }
    status_code, response, elapsed_time = make_request(BASE_URL, "post", "/v1/ask/", data)
    if response is None:
        raise Exception("Response is None")
    
    error = None
    if status_code != 200:
        error = response.get("data") or response

    log_result("post", "/v1/ask", elapsed_time, error)
    if error:
        raise Exception(error)
    
    return response

def main_function_test_langgpt():
    test_post_ask()