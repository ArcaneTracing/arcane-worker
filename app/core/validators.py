"""
Common validation utilities for the application.
Centralizes configuration and input validation logic.
"""
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass


def validate_required_field(
    value: Any,
    field_name: str,
    context: Optional[str] = None
) -> None:
    """
    Validate that a required field is present and not None/empty.
    
    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)
        context: Optional context information (e.g., "model configuration")
        
    Raises:
        ValidationError: If the field is missing or empty
    """
    if value is None or value == "":
        context_str = f" in {context}" if context else ""
        error_msg = f"{field_name} is required{context_str}"
        logger.error(error_msg)
        raise ValidationError(error_msg)


def validate_config_structure(
    config: Dict[str, Any],
    required_keys: List[str],
    config_name: str = "configuration"
) -> None:
    """
    Validate that a configuration dictionary has all required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names
        config_name: Name of the configuration (for error messages)
        
    Raises:
        ValidationError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]
    if missing_keys:
        error_msg = f"{config_name} missing required keys: {', '.join(missing_keys)}"
        logger.error(f"{error_msg}. Available keys: {list(config.keys())}")
        raise ValidationError(error_msg)


def extract_config_section(
    config: Dict[str, Any],
    section_key: str,
    section_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and validate a configuration section.
    
    Args:
        config: Main configuration dictionary
        section_key: Key of the section to extract
        section_name: Optional name for error messages
        
    Returns:
        Configuration section dictionary
        
    Raises:
        ValidationError: If the section is missing
    """
    section_name = section_name or section_key
    section = config.get(section_key)
    
    if not section:
        error_msg = f"{section_name} missing '{section_key}' key" if section_name != section_key else f"Configuration missing '{section_key}' key"
        logger.error(f"{error_msg}. Available keys: {list(config.keys())}")
        raise ValidationError(error_msg)
    
    if not isinstance(section, dict):
        error_msg = f"'{section_key}' must be a dictionary"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    return section

