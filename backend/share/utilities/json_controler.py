import json

from utilities.log_controler import LogControler
log_controler = LogControler()

def data_base_model_to_string(data):
    return data.json()

def data_string_or_dict_to_json_dump(data):
    return json.dumps(data)

def data_string_to_json_load(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # # Handle the case where the string cannot be converted to a dictionary
        # log_controler.log_error("Error converting string to dictionary", "data_string_to_json_load")
        return None
