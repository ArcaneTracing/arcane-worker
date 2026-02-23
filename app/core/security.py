"""
Security utilities for input validation and safe parsing.
Centralizes security-related validation logic.
"""
import json
import logging
from typing import Any, Optional
from pydantic_core import from_json

logger = logging.getLogger(__name__)

# Security constants
MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB - maximum size for JSON strings
MAX_STRING_LENGTH = 100 * 1024 * 1024  # 100MB - maximum string length
MAX_DICT_KEYS = 10000  # Maximum number of keys in a dictionary
MAX_LIST_LENGTH = 100000  # Maximum length of a list


class SecurityError(ValueError):
    """Custom exception for security-related errors."""
    pass


def validate_json_size(json_string: str, max_size: Optional[int] = None) -> None:
    """
    Validate that a JSON string is within size limits.
    
    Prevents DoS attacks from extremely large JSON payloads.
    
    Args:
        json_string: The JSON string to validate
        max_size: Maximum allowed size in bytes (default: MAX_JSON_SIZE)
        
    Raises:
        SecurityError: If JSON string exceeds size limit
    """
    max_size = max_size or MAX_JSON_SIZE
    actual_size = len(json_string.encode('utf-8'))
    
    if actual_size > max_size:
        error_msg = f"JSON string size ({actual_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
        logger.warning(error_msg)
        raise SecurityError(error_msg)


def safe_json_loads(
    json_string: str,
    max_size: Optional[int] = None,
    fallback: Any = None
) -> Any:
    """
    Safely parse JSON string with size validation and error handling.
    
    Args:
        json_string: The JSON string to parse
        max_size: Maximum allowed size in bytes (default: MAX_JSON_SIZE)
        fallback: Value to return if parsing fails (default: None)
        
    Returns:
        Parsed JSON object, or fallback value if parsing fails
    """
    if not isinstance(json_string, str):
        return json_string
    
    try:
        # Validate size before parsing
        validate_json_size(json_string, max_size)
        
        # Parse JSON
        return json.loads(json_string)
    except (json.JSONDecodeError, SecurityError) as e:
        logger.debug(f"JSON parsing failed: {e}")
        return fallback


def safe_from_json(
    json_string: str,
    allow_partial: bool = False,
    max_size: Optional[int] = None,
    fallback: Any = None
) -> Any:
    """
    Safely parse JSON string using Pydantic's from_json with size validation.
    
    Args:
        json_string: The JSON string to parse
        allow_partial: Whether to allow partial JSON (for LLM outputs)
        max_size: Maximum allowed size in bytes (default: MAX_JSON_SIZE)
        fallback: Value to return if parsing fails (default: None)
        
    Returns:
        Parsed JSON object, or fallback value if parsing fails
    """
    if not isinstance(json_string, str):
        return json_string
    
    try:
        # Validate size before parsing
        validate_json_size(json_string, max_size)
        
        # Parse using Pydantic's from_json
        return from_json(json_string, allow_partial=allow_partial)
    except (ValueError, TypeError) as e:
        logger.debug(f"Pydantic JSON parsing failed: {e}")
        return fallback


def validate_string_length(value: str, max_length: Optional[int] = None) -> None:
    """
    Validate that a string is within length limits.
    
    Args:
        value: The string to validate
        max_length: Maximum allowed length (default: MAX_STRING_LENGTH)
        
    Raises:
        SecurityError: If string exceeds length limit
    """
    max_length = max_length or MAX_STRING_LENGTH
    
    if len(value) > max_length:
        error_msg = f"String length ({len(value)}) exceeds maximum allowed length ({max_length})"
        logger.warning(error_msg)
        raise SecurityError(error_msg)


def validate_dict_size(data: dict, max_keys: Optional[int] = None) -> None:
    """
    Validate that a dictionary is within size limits.
    
    Args:
        data: The dictionary to validate
        max_keys: Maximum allowed number of keys (default: MAX_DICT_KEYS)
        
    Raises:
        SecurityError: If dictionary exceeds size limit
    """
    max_keys = max_keys or MAX_DICT_KEYS
    
    if len(data) > max_keys:
        error_msg = f"Dictionary size ({len(data)} keys) exceeds maximum allowed size ({max_keys} keys)"
        logger.warning(error_msg)
        raise SecurityError(error_msg)


def validate_list_length(data: list, max_length: Optional[int] = None) -> None:
    """
    Validate that a list is within length limits.
    
    Args:
        data: The list to validate
        max_length: Maximum allowed length (default: MAX_LIST_LENGTH)
        
    Raises:
        SecurityError: If list exceeds length limit
    """
    max_length = max_length or MAX_LIST_LENGTH
    
    if len(data) > max_length:
        error_msg = f"List length ({len(data)}) exceeds maximum allowed length ({max_length})"
        logger.warning(error_msg)
        raise SecurityError(error_msg)


def sanitize_format_spec(format_spec: str) -> str:
    """
    Sanitize format specification to prevent code injection.
    
    Only allows safe format specifiers. Rejects format specs that could
    execute arbitrary code or access attributes.
    
    Args:
        format_spec: Format specification string (e.g., ".2f", ">10")
        
    Returns:
        Sanitized format specification
        
    Raises:
        SecurityError: If format spec contains potentially unsafe characters
    """
    # Allow only safe format specifiers: digits, ., -, +, space, <, >, =, ^
    # Note: '.' is allowed for decimal precision (e.g., ".2f")
    # Block attribute access (!), function calls (), brackets [], braces {}, etc.
    unsafe_chars = {'!', '(', ')', '[', ']', '{', '}', '?', '@', '|', '&'}
    
    if any(char in format_spec for char in unsafe_chars):
        error_msg = f"Unsafe format specifier detected: {format_spec}"
        logger.warning(error_msg)
        raise SecurityError(error_msg)
    
    return format_spec

