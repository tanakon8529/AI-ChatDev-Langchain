from settings.configs import HOST, PORT_FASTAPI_AI_CHAT
from utilitys.http_request import make_request, log_result

import ssl
print(ssl.OPENSSL_VERSION)


# Assuming these are set up as environment variables or constants
if HOST == "api.apthai.com":
    BASE_URL = f"http://localhost:{PORT_FASTAPI_AI_CHAT}/langgpt"
else:
    BASE_URL = f"http://{HOST}:{PORT_FASTAPI_AI_CHAT}/langgpt"

def test_post_ask():
    """Test for asking a question."""
    method = "post"
    path = "/v1/ask/"
    # data = {
    #     "question": "ผลการประเมินความพึงพอใจของลูกค้าที่มีต่อ Call center สามปีย้อนหลัง กี่เปอร์เซ็น"
    # }
    # data = {
    #     "question": "รายชืื่อคณะกรรมการกำกับดูแลกิจการและการพัฒนาอย่างยั่งยืน มีใครบ้างและตำแหน่งอะไร"
    # }
    data = {
        "question": "มีบ้านหรือห้องที่ทำการขายแล้วและรอส่งมอบให้ลูกค้า ที่ยัง Active อยู่กี่โครงการ แบรนดอะไร มูลเท่าไหร่บ้างและรวมเป็นมูลค่าเท่าไหร่"
    }
    
    response_pack = []
    # Lopp post test 10 times
    for i in range(10):
        status_code, response, elapsed_time = make_request(BASE_URL, method, path, data)
        response_pack.append((status_code, response, elapsed_time))
        print(f"method: {method} | path: {path} | status_code: {status_code} | elapsed_time: {elapsed_time:.2f} seconds")

    response_error_pack = [response for response in response_pack if response[0] != 200]
    if response_error_pack:
        log_result(method, path, sum([response[2] for response in response_pack]) / len(response_pack), response_error_pack)
        raise Exception("Failed test.")

    log_result(method, path, sum([response[2] for response in response_pack]))
    return response_pack


def main_function_test_langgpt():
    test_post_ask()