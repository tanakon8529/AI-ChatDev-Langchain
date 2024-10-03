import secrets
import random
import string

from utilities.redis_connector import get_client
from utilities.time_controler import get_current_date
from utilities.log_controler import LogControler
from utilities.json_controler import data_string_to_json_load, data_string_or_dict_to_json_dump
log_controler = LogControler()

def generate_cookie():
    return secrets.token_urlsafe()

# generate_RegRefCode gen char required have 3 type this.. upper, lower, digit with length 8
def generate_RegRefCode(length=8):
    if length < 3:
        raise ValueError("Length must be at least 3")
    uppercase_letters = string.ascii_uppercase
    lowercase_letters = string.ascii_lowercase
    digits = string.digits
    mandatory_chars = [random.choice(uppercase_letters), random.choice(lowercase_letters), random.choice(digits)]
    rest_length = length - 3
    all_chars = uppercase_letters + lowercase_letters + digits
    rest_chars = random.choices(all_chars, k=rest_length)
    code_chars = mandatory_chars + rest_chars
    random.shuffle(code_chars)
    return ''.join(code_chars)


class CookieControler:
    def __init__(self):
        self.redis = get_client()
        self.session_expire_in_3_month = 7776000

    # if same Username in redis more than 5 delete all, for security
    def check_and_delete_cookie_session_by_Username(self, data):
        result = {"error_server": "01", "msg": "Error delete_cookie_session_by_Username something wrong"}
        try:
            Username = data.Username.lower()  # Convert to lowercase for case-insensitive comparison
            # First: check if the same Username in redis more than 5
            keys = self.redis.keys()
            count = 0
            for key in keys:
                value = data_string_to_json_load(self.redis.get(key))
                if not value:
                    continue

                Redis_Username = value.get("Username", "").lower()  # Convert to lowercase
                if Redis_Username == Username:
                    count += 1
            
            # Second: if more than 5, delete all by Username
            if count > 5:
                for key in keys:
                    value = data_string_to_json_load(self.redis.get(key))
                    if not value:
                        continue

                    Redis_Username = value.get("Username", "").lower()  # Convert to lowercase
                    if Redis_Username == Username:
                        self.redis.delete(key)

                result = {"error_server": "00", "msg": f"{Username} more than 5 devices, Please login again"}
            else:
                result = {"detail": "Username not more than 5"}
        except Exception as e:
            log_controler.log_error(str(e), "check_and_delete_cookie_session_by_Username")
            result = {"error_server": "01", "msg": e}
        return result
        
    # session login expire in 90 days
    def fill_cookie_session(self, data):
        result = {"error_server": "01", "msg": "Error fill_cookie_session something wrong"}
        try:
            # security check, if same Username in redis more than 10 delete all
            result_security_check = self.check_and_delete_cookie_session_by_Username(data)
            if "error_server" in result_security_check:
                return result_security_check

            key = generate_cookie()
            value = {
                "SessionToken": key,
                "SessionIP": data.SessionIP,
                "DeviceName": data.DeviceName,
                "Username": data.Username,
                "AddOn": get_current_date().isoformat()  # convert datetime to ISO format string
            }
            
            data = data_string_or_dict_to_json_dump(value)
            self.redis.setex(key, self.session_expire_in_3_month, data)
            result = value
        except Exception as e:
            log_controler.log_error(str(e), "fill_cookie_session")
            result = {"error_server": "01","msg": e}
        return result

    def get_cookie_session_by_SessionToken_and_Username_not_SessionExpireIn_and_Check_IP_Device(self, data):
        result = {"error_server": "01", "msg": "Error get_cookie_session_by_SessionToken_and_Username_not_SessionExpireIn_and_Check_IP_Device something wrong"}
        try:
            key = data.SessionToken
            data_redis = self.redis.get(key)
            if data_redis is None:
                result = {"error_server": "01", "msg": "SessionToken not found"}
            else:
                value = data_string_to_json_load(data_redis)
                if value["Username"] == data.Username and value["SessionIP"] == data.SessionIP and value["DeviceName"] == data.DeviceName:
                    result = value
                else:
                    result = {"error_server": "01", "msg": "SessionToken not match"}
        except Exception as e:
            log_controler.log_error(str(e), "get_cookie_session_by_SessionToken_and_Username_not_SessionExpireIn_and_Check_IP_Device")
            result = {"error_server": "01","msg": e}
        return result