__all__ = ["setup_logger"]
import os
import logging

def setup_logger(log_path, log_name, file_name):
    """
    Sets up a logger with the specified log path, log name, and file name.

    Args:
        log_path (str): The path to the directory where the log file will be stored.
        log_name (str): The name of the logger.
        file_name (str): The name of the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger