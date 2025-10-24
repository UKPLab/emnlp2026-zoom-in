import logging
import sys
from pathlib import Path

# Global logger instance
_logger = None
_configured = None


def get_logger(name=None):
    """
    Get or create a logger instance

    Args:
        name (str): Logger name (usually __name__)
        log_file (str): Path to log file
        log_level: Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger, _configured

    # If logger doesn't exist yet, create a basic one
    if _logger is None:
        _logger = logging.getLogger('main')
        # Don't configure handlers yet - wait for setup_project_logging

    return _logger if name is None else _logger.getChild(name)


def setup_project_logging(log_file=None, log_level=logging.INFO):
    """
    Configure logging for the entire project
    This should be called once at runtime after you know the log file path

    Args:
        log_file (str, optional): Path to log file. If None, only logs to console
        log_level: Logging level
    """
    global _logger, _configured

    # If already configured, don't configure again
    if _configured:
        return _logger

    # Create logger if it doesn't exist
    if _logger is None:
        _logger = logging.getLogger('main')

    # Clear any existing handlers to avoid duplicates
    _logger.handlers.clear()

    # Set level
    _logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Always create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # Only create file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    # Mark as configured
    _configured = True

    return _logger
