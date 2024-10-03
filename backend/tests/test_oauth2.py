import requests
import time

from settings.configs import HOST, PORT_FASTAPI_OAUTH2, USERNAME_ADMIN, PASSWORD_ADMIN

class TestOAuth2:
    def __init__(self) -> None:
        # Assuming these are set up as environment variables or constants
        if HOST == "api.qfreethailand.com":
            self.BASE_URL = f"http://localhost:{PORT_FASTAPI_OAUTH2}/oauth"
        else:
            self.BASE_URL = f"http://{HOST}:{PORT_FASTAPI_OAUTH2}/oauth"

        self.CLIENT_ID = USERNAME_ADMIN
        self.CLIENT_SECRET = PASSWORD_ADMIN
        self.token = self.get_access_token().get('access_token')
        self.test_protected_route(self.token)

    def make_request(self, base_url, method, endpoint, data=None, files=None, is_json=True, headers=None):
        """
        Makes an HTTP request and returns the response and elapsed time.
        """
            
        url = f"{base_url}{endpoint}"
        start_time = time.time()
        try:
            if method == "get":
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
            status_code = response.status_code
            return status_code, response.json(), time.time() - start_time
        except Exception as e:
            return 500, {"error": str(e)}, time.time() - start_time
            
    def log_result(self, operation, endpoint, elapsed_time, error=None):
        """
        Logs the result of an operation.
        """
        status = "FAILED" if error else "PASS"
        error_msg = f" | error: {error}" if error else ""
        print(f"{status} | route: {operation} {endpoint} | elapsed time: {elapsed_time:.2f} seconds{error_msg}")

    def get_access_token(self):
        """Function to obtain an access token."""
        headers = {
            'client-id': self.CLIENT_ID,
            'client-secret': self.CLIENT_SECRET,
        }
        status_code, response, elapsed_time = self.make_request(self.BASE_URL, "post", "/v1/token/", headers=headers)
        if response is None:
            raise Exception("Response is None")
        
        error = None
        if status_code != 200:
            error = response.get("detail") or response

        self.log_result("post", "/v1/token", elapsed_time, error)
        if error:
            raise Exception(error)
        
        return response

    def test_protected_route(self, token):
        """Test for accessing a protected route."""
        headers = {'token': f"Bearer {token}"}
        status_code, response, elapsed_time = self.make_request(self.BASE_URL, "get", "/v1/protected/", headers=headers)
        if response is None:
            raise Exception("Response is None")
        
        error = None
        if status_code != 200:
            error = response.get("detail") or response

        self.log_result("get", "/v1/protected", elapsed_time, error)
        if error:
            raise Exception(error)
        
        return response

    def main_function_test_oauth2(self):
        return self.token
