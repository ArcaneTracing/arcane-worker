"""
Unit tests for logging configuration.
"""
import pytest
import logging
from app.core.logging_config import setup_logging


class TestLoggingConfig:
    """Tests for logging configuration."""
    
    def test_setup_logging_default_level(self):
        """Should setup logging with default INFO level."""
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO
    
    def test_setup_logging_custom_level(self):
        """Should setup logging with custom level."""
        setup_logging(level="DEBUG")
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.DEBUG
    
    def test_setup_logging_warning_level(self):
        """Should setup logging with WARNING level."""
        setup_logging(level="WARNING")
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.WARNING
    
    def test_setup_logging_error_level(self):
        """Should setup logging with ERROR level."""
        setup_logging(level="ERROR")
        
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.ERROR
    
    def test_creates_console_handler(self):
        """Should create console handler."""
        setup_logging()
        
        root_logger = logging.getLogger()
        handlers = root_logger.handlers
        assert len(handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)

