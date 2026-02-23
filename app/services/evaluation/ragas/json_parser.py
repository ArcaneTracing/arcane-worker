"""
JSON parsing utilities for RAGAS evaluation.
Handles parsing of JSON strings in score_mapping data.
"""
from typing import Any
from app.core.security import safe_from_json


def _looks_like_json(stripped: str) -> bool:
    """Check if a string looks like it could be JSON."""
    if not stripped:
        return False
    return (
        stripped.startswith(("[", "{", '"')) or
        stripped.lower() in ("true", "false", "null") or
        (stripped[0].isdigit() or stripped[0] == "-")
    )


def _extract_incomplete_array_content(stripped: str) -> list[str] | None:
    """
    Extract content from an incomplete JSON array string.
    
    Handles cases like '["Albert Einstein was born on March 14' -> extract the string.
    """
    if len(stripped) <= 2:  # Just '[' or '[]'
        return None
    
    content = stripped[1:].strip()  # Content after '['
    if not content.startswith('"'):
        return None
    
    # Extract the string value (even if incomplete - no closing quote)
    string_content = content[1:]  # Remove opening quote
    end_quote_idx = string_content.find('"')
    
    if end_quote_idx >= 0:
        extracted = string_content[:end_quote_idx]
    else:
        # No closing quote found - use everything (incomplete string)
        extracted = string_content.rstrip(']').rstrip(',').strip()
    
    return [extracted] if extracted else None


def _parse_json_string(data: str) -> Any:
    """
    Parse a JSON string using Pydantic's partial JSON parsing.
    
    Handles incomplete JSON strings, which is particularly useful for LLM outputs.
    Reference: https://docs.pydantic.dev/latest/concepts/json/#partial-json-parsing
    """
    stripped = data.strip()
    
    if not _looks_like_json(stripped):
        return data
    
    try:
        # Use Pydantic's partial JSON parsing to handle incomplete JSON
        # Security: Uses safe_from_json with size validation
        parsed = safe_from_json(data, allow_partial=True, fallback=None)
        
        if parsed is None or parsed == data:
            return data
        
        # Handle incomplete array parsing
        if stripped.startswith("[") and isinstance(parsed, list) and len(parsed) == 0:
            extracted = _extract_incomplete_array_content(stripped)
            if extracted is not None:
                return extracted
        
        # Recursively parse the parsed result in case it contains nested JSON strings
        return parse_json_strings(parsed)
    except (ValueError, TypeError):
        # If parsing fails, return the original string
        return data


def parse_json_strings(data: Any) -> Any:
    """
    Recursively parse JSON strings in the data structure using Pydantic's partial JSON parsing.
    
    This function handles cases where values in score_mapping are JSON strings
    (including partial JSON) that need to be parsed into their proper Python types.
    
    Uses Pydantic's partial JSON parsing feature (pydantic_core.from_json with allow_partial=True)
    to handle incomplete JSON strings, which is particularly useful for LLM outputs.
    
    Attempts to parse any string that could be JSON (not just lists and dicts, but also
    strings, numbers, booleans, null, etc.) using Pydantic's from_json parser.
    
    Reference: https://docs.pydantic.dev/latest/concepts/json/#partial-json-parsing
    
    Args:
        data: The data to parse (can be dict, list, str, or other types)
        
    Returns:
        Parsed data with JSON strings converted to their Python equivalents
    """
    if isinstance(data, dict):
        return {key: parse_json_strings(value) for key, value in data.items()}
    
    if isinstance(data, list):
        return [parse_json_strings(item) for item in data]
    
    if isinstance(data, str):
        return _parse_json_string(data)
    
    return data

