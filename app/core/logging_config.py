"""
Logging configuration for the application.
Ensures logging works across all threads including background scheduler threads.
"""
import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = None):
    """
    Configure logging for the application.
    This ensures logging works in both main thread and background threads.
    
    Args:
        level: Logging level (default: INFO)
    """
    if level is None:
        level = "INFO"
    
    # Create a formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to prevent duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Ensure loggers propagate to root (important for background threads)
    # Note: root logger doesn't propagate (it's the top level), but child loggers should
    
    # Set level for our application loggers
    app_logger = logging.getLogger("app")
    app_logger.setLevel(getattr(logging, level.upper()))
    app_logger.propagate = True
    
    # Set level for worker loggers
    worker_logger = logging.getLogger("app.worker")
    worker_logger.setLevel(getattr(logging, level.upper()))
    worker_logger.propagate = True
    
    # Set level for experiment service loggers
    experiment_logger = logging.getLogger("app.services.experiment")
    experiment_logger.setLevel(getattr(logging, level.upper()))
    experiment_logger.propagate = True

