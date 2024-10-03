import secrets
from datetime import datetime, timedelta

def if_none_to_current_date(date_check_none):
    current_date = datetime.now()
    if date_check_none == None:
        return current_date
    return date_check_none

def get_current_date():
    return datetime.now()

def get_expire_date(days):
    return datetime.now() + timedelta(days=days)

def get_previous_date(days):
    return datetime.now() - timedelta(days=days)

def convert_iso_to_datetime(iso_date):
    return datetime.fromisoformat(iso_date)

# date string format: 'YYYY-MM-DD' to datetime
def convert_date_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

def generate_unique_id(length=8):
    """Generate a unique ID combining a timestamp and a random string."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = secrets.token_hex(length // 2)  # Generate a random hex string
    return f"{timestamp}_{random_str}"

def if_not_more_than_current_date_fail(date_check):
    current_date = datetime.now()
    if date_check > current_date:
        return True
    return False