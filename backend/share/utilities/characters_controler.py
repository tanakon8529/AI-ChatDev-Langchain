import re

class TextFormatter:
    def remove_spaces(self, text):
        """Remove all spaces from the text."""
        return text.replace(" ", "")

    def remove_special_characters(self, text):
        """Remove all special characters from the text, keeping only letters and numbers."""
        return re.sub(r"[^\w\s]", "", text)

    def remove_numbers(self, text):
        """Remove all numbers from the text."""
        return re.sub(r"\d", "", text)

    def check_phone_format(self, phone_number):
        """Check if the phone number matches the +66xxxxxxxxx format."""
        pattern = r"^\+66\d{9}$"
        return bool(re.match(pattern, phone_number))
    
    def replace_phone_format(self, phone_number):
        """Replace the phone number with +66xxxxxxxxx format."""
        pattern = r"0(\d{9})"
        return re.sub(pattern, r"+66\1", phone_number)
