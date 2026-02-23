"""
Utilities for extracting output and tool calls from Google GenAI GenerateContentResponse.

The SDK's response.text only returns text parts and logs a warning when there are
function_call parts. We iterate over candidates[0].content.parts to extract both
text and function_call parts correctly.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Tuple


def _parse_args(args: Any) -> Dict[str, Any]:
    """Parse function call args to dict. Handles dict or JSON string."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        from app.core.security import safe_json_loads
        return safe_json_loads(args, fallback={})
    return {}


def extract_output_and_tool_calls(response) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Extract output text and tool calls from GenerateContentResponse.

    Iterates over candidates[0].content.parts to handle both text and function_call parts.
    Avoids the SDK warning about non-text parts.

    Returns:
        (output_text, tool_calls) - tool_calls is None if not present, else list of
        ToolCall dicts (id, name, arguments).
    """
    if not response or not getattr(response, "candidates", None):
        return "", None

    candidates = response.candidates
    if not candidates:
        return "", None

    content = getattr(candidates[0], "content", None)
    if not content:
        return "", None

    parts = getattr(content, "parts", None) or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for part in parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        elif hasattr(part, "function_call") and part.function_call:
            # log fc as json
            print(f"fc: part.function_call: {part.function_call}")
            fc = part.function_call
            name = getattr(fc, "name", None) or ""
            call_id = getattr(fc, "id", None) or ""
            arguments = _parse_args(getattr(fc, "args", None))
            tool_calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )

    output = "".join(text_parts) if text_parts else ""
    return output, tool_calls if tool_calls else None
