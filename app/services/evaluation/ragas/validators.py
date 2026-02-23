"""
Common validators for RAGAS evaluation inputs.
"""
from typing import List, Any


def convert_to_string(v: str | int | float | None) -> str:
    """
    Convert value to string (handles int, float, None, etc.).
    
    Args:
        v: Value to convert (str, int, float, None, etc.)
        
    Returns:
        String representation of the value, or empty string if None
    """
    if v is None:
        return ""
    return str(v)


def convert_string_to_list(v: str | List[str]) -> List[str]:
    """
    Convert single string to single-item list if needed.
    
    Args:
        v: String or list of strings
        
    Returns:
        List of strings (single-item list if input was a string)
    """
    if isinstance(v, str):
        return [v]
    return v


def convert_list_to_string(v: str | List[str]) -> str:
    """
    Convert list to string (take first element) or preserve string.
    
    Used for metrics that expect a string but may receive a list.
    
    Args:
        v: String or list of strings
        
    Returns:
        String (first element if input was a list, or original string)
    """
    if isinstance(v, list):
        return v[0] if len(v) > 0 else ""
    return str(v) if v is not None else ""

