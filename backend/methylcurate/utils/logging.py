__all__ = ["setup_logger"]
import logging
import os


def setup_logger(log_path, log_name, file_name, *, thread_id: str | None = None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    file_handler.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(name)s - %(levelname)s"
    if thread_id:
        fmt += f" - [thread:{thread_id}]"
    fmt += " - %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger
