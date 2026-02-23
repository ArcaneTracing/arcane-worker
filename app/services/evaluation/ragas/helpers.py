"""
Helper functions for RAGAS evaluation metrics.
"""
from typing import Any


def extract_score(result: Any) -> int | float:
    """
    Extract score from RAGAS evaluation result.
    
    Handles both float/int results and objects with .value attribute.
    
    Args:
        result: RAGAS evaluation result (can be float/int or object with .value)
        
    Returns:
        Numeric score (int or float)
    """
    if isinstance(result, (int, float)):
        return result
    return result.value

