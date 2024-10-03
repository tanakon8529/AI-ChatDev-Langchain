from argon2 import PasswordHasher
# from utilities.log_controler import LogControler
# log_controler = LogControler()

def hash_password(password):
    """
    Hash a password for storing.
    """
    ph = PasswordHasher(hash_len=32)
    return ph.hash(password)

def verify_password(hashed_password, password):
    """
    Verify a stored password against one provided by user
    """
    ph = PasswordHasher()
    try:
        ph.verify(hashed_password, password)
        return True
    except Exception as e:
        # log_controler.log_error(str(e), "verify_password")
        return False