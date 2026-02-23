"""
Unit tests for error handling utilities.
"""
import pytest
from unittest.mock import Mock
from app.core.error_handling import (
    extract_error_details,
    log_api_error
)


class TestExtractErrorDetails:
    """Tests for extract_error_details function."""
    
    def test_extracts_basic_error(self):
        """Should extract basic error information."""
        error = ValueError("Test error")
        details = extract_error_details(error)
        
        assert details["error_type"] == "ValueError"
        assert details["error_message"] == "Test error"
        assert details["error_details"] == "Test error"
    
    def test_extracts_error_with_response(self):
        """Should extract error with response object."""
        error = ValueError("API error")
        error.response = Mock()
        error.response.json = Mock(return_value={"error": "Bad request"})
        
        details = extract_error_details(error)
        
        assert details["error_type"] == "ValueError"
        assert "API Error Body" in details["error_details"]
    
    def test_handles_response_without_json(self):
        """Should handle response without json method."""
        error = ValueError("API error")
        error.response = Mock()
        error.response.text = "Error text"
        del error.response.json  # Remove json method
        
        details = extract_error_details(error)
        
        assert "API Response Text" in details["error_details"] or "Error text" in details["error_details"]
    
    def test_handles_response_with_status_code(self):
        """Should handle response with status_code."""
        error = ValueError("API error")
        error.response = Mock()
        error.response.status_code = 500
        del error.response.json
        del error.response.text
        
        details = extract_error_details(error)
        
        assert "Status Code" in details["error_details"] or "500" in details["error_details"]


class TestLogApiError:
    """Tests for log_api_error function."""
    
    def test_logs_error_with_context(self, caplog):
        """Should log error with context information."""
        error = ValueError("Test error")
        
        log_api_error(
            error=error,
            service_name="TestService",
            context="test_method"
        )
        
        # Check that error was logged
        assert "TestService" in caplog.text
        assert "test_method" in caplog.text
    
    def test_logs_error_with_additional_info(self, caplog):
        """Should log error with additional information."""
        error = ValueError("Test error")
        
        log_api_error(
            error=error,
            service_name="TestService",
            additional_info={"model": "gpt-4", "version": "1.0"}
        )
        
        # Check that additional info was logged
        assert "model=gpt-4" in caplog.text or "gpt-4" in caplog.text
    
    def test_logs_without_context(self, caplog):
        """Should log error without context."""
        error = ValueError("Test error")
        
        log_api_error(
            error=error,
            service_name="TestService"
        )
        
        assert "TestService" in caplog.text

