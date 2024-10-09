# /utilities/nltk_handler.py

import os
import nltk
from utilities.log_controler import LogControler

class NLTKHandler:
    """
    Handles the initialization and management of NLTK resources.
    Ensures that necessary tokenizers are available, downloading them if required.
    """
    def __init__(self, log_controler: LogControler, data_dir: str = None):
        """
        Initializes the NLTKHandler.

        :param log_controler: An instance of LogControler for logging purposes.
        :param data_dir: Optional directory to store NLTK data. If not provided, defaults to NLTK's default path.
        """
        self.log_controler = log_controler
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'nltk_data')
        self.ensure_nltk_data_dir()
        self.initialize_punkt()

    def ensure_nltk_data_dir(self):
        """
        Ensures that the NLTK data directory exists and is included in NLTK's search path.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.log_controler.log_info(f"Created NLTK data directory at: {self.data_dir}")
        else:
            self.log_controler.log_info(f"NLTK data directory exists at: {self.data_dir}")

        # Append the data directory to NLTK's search path if not already present
        if self.data_dir not in nltk.data.path:
            nltk.data.path.append(self.data_dir)
            self.log_controler.log_info(f"Appended NLTK data directory to NLTK path: {self.data_dir}")

    def initialize_punkt(self):
        """
        Checks for the presence of the 'punkt' tokenizer and downloads it if not found.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            self.log_controler.log_info("NLTK 'punkt' tokenizer found.")
        except LookupError:
            self.log_controler.log_info("NLTK 'punkt' tokenizer not found. Downloading...")
            try:
                nltk.download('punkt_tab', download_dir=self.data_dir)
                self.log_controler.log_info("NLTK 'punkt' tokenizer downloaded successfully.")
            except Exception as e:
                self.log_controler.log_error(f"Failed to download NLTK 'punkt' tokenizer: {e}", "NLTKHandler")
                raise
