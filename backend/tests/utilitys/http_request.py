
import requests
import time
import json
from test_oauth2 import TestOAuth2
main_function_test_oauth2 = TestOAuth2().main_function_test_oauth2

# Utility Functions
def make_request(base_url, method, endpoint, data=None, files=None, is_json=True):
    """
    Makes an HTTP request and returns the response and elapsed time.
    """
    token = main_function_test_oauth2()
    headers = {'token': f"Bearer {token}"} if token else {}
    url = f"{base_url}{endpoint}"
    start_time = time.time()
    try:
        if method == "get":
            if is_json and data:
                data = json.dumps(data)
                response = requests.request("GET", url, headers=headers, data=data)
            else:
                response = requests.get(url, headers=headers)
        elif method == "post":
            response = requests.post(url, headers=headers, json=data) if is_json else requests.post(url, headers=headers, data=data, files=files)
        elif method == "put":
            response = requests.put(url, headers=headers, json=data) if is_json else requests.put(url, headers=headers, data=data)
        elif method == "delete":
            response = requests.delete(url, headers=headers, json=data)
        else:
            raise ValueError("Invalid HTTP method.")
        
        response.raise_for_status()
        return response.status_code, response.json(), time.time() - start_time
    except requests.exceptions.HTTPError as http_err:
        error = response.json()
        message_error = error["message_error_test"] if http_err in error else ""
        return response.status_code, message_error, time.time() - start_time
    except Exception as e:
        error = response.json()
        message_error = error["message_error_test"] if str(e) in error else ""
        return 500, message_error, time.time() - start_time
        
def log_result(operation, endpoint, elapsed_time, error=None):
    """
    Logs the result of an operation.
    """
    status = "FAILED" if error else "PASS"
    error_msg = f" | error: {error}" if error else ""
    print(f"{status} | route: {operation} {endpoint} | elapsed time: {elapsed_time:.2f} seconds{error_msg}")
