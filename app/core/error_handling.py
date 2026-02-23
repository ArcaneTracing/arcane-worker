"""
Common error handling utilities for the application.
Centralizes error extraction and logging patterns.
"""
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def extract_error_details(error: Exception) -> Dict[str, Any]:
    """
    Extract detailed error information from an exception.
    
    Handles various error types including API errors with response objects.
    
    Args:
        error: The exception to extract details from
        
    Returns:
        Dictionary containing error_type, error_message, and error_details
    """
    error_type = type(error).__name__
    error_msg = str(error)
    error_details = error_msg
    
    # Try to extract API error details (OpenAI, Anthropic, etc.)
    if hasattr(error, 'response') and error.response is not None:
        try:
            if hasattr(error.response, 'json'):
                error_body = error.response.json()
                error_details = f"{error_msg} | API Error Body: {error_body}"
            elif hasattr(error.response, 'text'):
                error_details = f"{error_msg} | API Response Text: {error.response.text}"
            elif hasattr(error.response, 'status_code'):
                error_details = f"{error_msg} | Status Code: {error.response.status_code}"
        except Exception as parse_error:
            logger.debug(f"Could not parse error response: {parse_error}")
    
    return {
        "error_type": error_type,
        "error_message": error_msg,
        "error_details": error_details
    }


def log_api_error(
    error: Exception,
    service_name: str,
    context: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log API errors with detailed information.
    
    Args:
        error: The exception that occurred
        service_name: Name of the service where error occurred (e.g., "OpenAIModelService")
        context: Optional context information (e.g., "execute()")
        additional_info: Optional dictionary with additional context (e.g., model name)
    """
    error_info = extract_error_details(error)
    
    context_str = f" in {context}" if context else ""
    additional_str = ""
    if additional_info:
        parts = [f"{k}={v}" for k, v in additional_info.items()]
        additional_str = f" - {', '.join(parts)}"
    
    logger.error(
        f"API error in {service_name}{context_str} - "
        f"Error type: {error_info['error_type']}, "
        f"Error details: {error_info['error_details']}{additional_str}",
        exc_info=True
    )

