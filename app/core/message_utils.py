"""
Common utilities for message conversion and content extraction.
Used across different LLM service implementations.
"""
from __future__ import annotations

from typing import List, Dict, Union, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.models.schemas import Tools

from app.models.schemas import ContentPart, PromptMessage


def extract_text_from_content(content: Union[str, List[Union[Dict[str, Any], ContentPart]]]) -> str:
    """
    Extract text from content which can be a string or list of ContentParts.
    
    This is a common pattern used across multiple LLM services.
    
    Args:
        content: Content that can be a string or list of ContentPart objects/dicts
        
    Returns:
        Extracted text as a single string
    """
    if isinstance(content, str):
        return content
    
    if not isinstance(content, list):
        return ""
    
    # Extract text from each ContentPart
    text_parts = []
    for part in content:
        if isinstance(part, dict) and 'text' in part:
            text_parts.append(part['text'])
        elif hasattr(part, 'text'):
            text_parts.append(part.text)
    
    return ' '.join(text_parts) if text_parts else ""


def normalize_role(role: str, supported_roles: Optional[set[str]] = None) -> str:
    """
    Normalize role to supported roles.
    
    Maps unknown roles to default roles based on common patterns.
    
    Args:
        role: The role to normalize
        supported_roles: Set of supported roles (default: {"system", "user", "assistant"})
        
    Returns:
        Normalized role
    """
    if supported_roles is None:
        supported_roles = {"system", "user", "assistant"}
    
    if role in supported_roles:
        return role
    
    # Map unknown roles: anything that's not assistant becomes user
    return "assistant" if role == "assistant" else "user"


def convert_tools_to_format(
    tools: Optional["Tools"],
    format_type: str = "openai"
) -> Optional[List[Dict[str, Any]]]:
    """
    Pass-through tools. User provides provider-specific format; no conversion.
    format_type ignored for backwards compatibility.
    """
    if not tools or not tools.tools:
        return None
    return tools.tools

