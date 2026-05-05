import os
import sys
import logging
import traceback
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_task_logger(module_name, log_path=None, task_id=None, console=True):
    """
    Create a module-level task logger.

    Parameters:
    - module_name: Name of the current module, such as search, download, normalize, pca, model.
    - log_path: Directory used to save log files.
    - task_id: Optional task id. If None, timestamp will be used.
    - console: Whether to also print logs to console.

    Returns:
    - logger: logging.Logger object.
    - log_file: absolute path of the generated log file.
    """
    if task_id is None:
        task_id = get_timestamp()

    if log_path is None:
        log_path = os.path.join(os.getcwd(), "logs")

    log_path = os.path.abspath(log_path)
    ensure_dir(log_path)

    log_file = os.path.join(log_path, f"{module_name}_{task_id}.log")

    logger_name = f"{module_name}_{task_id}_{id(log_file)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("=" * 100)
    logger.info(f"Logger created for module: {module_name}")
    logger.info(f"Task id: {task_id}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 100)

    return logger, log_file


def log_section(logger, title):
    logger.info("")
    logger.info("=" * 100)
    logger.info(title)
    logger.info("=" * 100)


def log_args(logger, **kwargs):
    logger.info("Input arguments:")
    for key, value in kwargs.items():
        logger.info(f"  - {key}: {value}")


def log_path_status(logger, path_name, path_value):
    abs_path = os.path.abspath(path_value) if path_value else path_value
    exists = os.path.exists(abs_path) if abs_path else False
    logger.info(f"{path_name}: {abs_path}")
    logger.info(f"{path_name} exists: {exists}")


def log_exception(logger, message, exc):
    logger.error(message)
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {exc}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())


def safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return "unknown"