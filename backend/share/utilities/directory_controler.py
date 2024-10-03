import os

def check_os_system():
    """Check if the operating system is Windows."""
    return os.name == 'nt'  # Returns True if Windows, False otherwise

def create_directory(path):
    """Create a directory if it doesn't exist and return True if created, False otherwise."""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def get_real_path(path):
    """Get the real path of a file or directory, ensuring correct path separators."""
    real_path = os.path.realpath(path)
    if check_os_system():
        real_path = real_path.replace('\\', '/')
    
    # Only create directory if the path does not have a file extension
    if not os.path.splitext(real_path)[1]:  
        create_directory(real_path)
    
    return real_path

def remove_name_directory_for_download(file_path):
    """Remove the name app/ of the directory from the file path."""
    return file_path.replace('app/', '')