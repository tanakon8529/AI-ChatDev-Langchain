import inspect, os
from loguru import logger

from settings.configs import set_microservice_name_by_api_path

class LogControler:
    def __init__(self, base_log_dir: str = f"logs", **kwargs):
        self.microservice_name = set_microservice_name_by_api_path(kwargs.get("port", ""))
        self.base_log_dir = f"./{self.microservice_name}/{base_log_dir}"
        self.setup_logger()

    def setup_logger(self):
        # Format for the log messages
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[path_file]}</cyan> | <cyan>{extra[function_name]}</cyan> | <red>{extra[error]}</red> | <level>{message}</level>"

        # Ensures a new file is created daily and includes the time in the filename
        filename = self.base_log_dir + "/{time:YYYYMMDD}.log"

        # Adding the logger with necessary configurations
        logger.add(filename, format=log_format, rotation="1 day", encoding='utf-8', enqueue=True, diagnose=True)

    def get_caller_info(self):
        """
        Retrieves the caller's file path and function name using the inspect module.
        """
        path_file = "Unknown"
        function_name = "Unknown"
        try:
            # Get the frame of the caller to the function that called get_caller_info
            frame = inspect.currentframe().f_back.f_back
            path_file = frame.f_globals["__file__"]
            function_name = frame.f_code.co_name
        except Exception as e:
            logger.error("Error in get_caller_info: {}".format(str(e)))

        return path_file, function_name

    def log_debug(self, message: str, data_debug: any = None):
        path_file, function_name = self.get_caller_info()
        # Bind the necessary keys before logging
        bound_logger = logger.bind(path_file=path_file, function_name=function_name, error="None")
        if data_debug is not None:
            # Log the debug data with the bound context
            bound_logger.debug(data_debug)
        # Log the primary message with the bound context
        bound_logger.debug(message)
        
    def log_error(self, error: str, message: str):
        path_file, function_name = self.get_caller_info()
        logger.bind(path_file=path_file, function_name=function_name, error=error).error(message)
