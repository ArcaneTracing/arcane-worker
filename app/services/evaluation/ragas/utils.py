"""
Utility functions for RAGAS evaluation.
"""
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall


class ConversationMessage(BaseModel):
    """Pydantic model for conversation messages."""
    role: Literal["system", "user", "assistant", "tool", "human", "ai"] = Field(..., description="Message role")
    content: str = Field(default="", description="Message content")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID (for assistant messages)")
    tool_call_function_name: Optional[str] = Field(None, description="Tool call function name (for assistant messages)")
    tool_call_function_arguments: Optional[str | Dict[str, Any]] = Field(None, description="Tool call function arguments (for assistant messages)")
    tool_calls: Optional[List[Dict[str, Any] | ToolCall]] = Field(None, description="List of tool calls (legacy format)")
    
    @field_validator('tool_call_function_arguments', mode='before')
    @classmethod
    def parse_json_arguments(cls, v):
        """Parse JSON string arguments to dict if needed (with security validation)."""
        from app.core.security import safe_json_loads
        
        if isinstance(v, str):
            return safe_json_loads(v, fallback={})
        return v
    
    def _parse_tool_call_arguments(self) -> Dict[str, Any]:
        """Parse tool call function arguments to a dictionary."""
        if not self.tool_call_function_arguments:
            return {}
        
        if isinstance(self.tool_call_function_arguments, dict):
            return self.tool_call_function_arguments
        
        if isinstance(self.tool_call_function_arguments, str):
            from app.core.security import safe_json_loads
            return safe_json_loads(self.tool_call_function_arguments, fallback={})
        
        return {}
    
    def _build_tool_calls_from_new_format(self) -> List[ToolCall]:
        """Build tool calls from new format (single tool call fields)."""
        if not (self.tool_call_id and self.tool_call_function_name):
            return []
        
        args = self._parse_tool_call_arguments()
        return [ToolCall(name=self.tool_call_function_name, args=args)]
    
    def _build_tool_calls_from_legacy_format(self) -> List[ToolCall]:
        """Build tool calls from legacy format (tool_calls list)."""
        if not self.tool_calls:
            return []
        
        ragas_tool_calls = []
        for tc in self.tool_calls:
            if isinstance(tc, ToolCall):
                ragas_tool_calls.append(tc)
            elif isinstance(tc, dict):
                ragas_tool_calls.append(ToolCall(
                    name=tc.get("name", ""),
                    args=tc.get("args", {})
                ))
        
        return ragas_tool_calls
    
    def _get_tool_calls(self) -> List[ToolCall] | None:
        """Get tool calls from either new or legacy format."""
        tool_calls = self._build_tool_calls_from_new_format()
        if tool_calls:
            return tool_calls
        
        tool_calls = self._build_tool_calls_from_legacy_format()
        return tool_calls if tool_calls else None
    
    def to_ragas_message(self) -> HumanMessage | AIMessage | ToolMessage:
        """Convert this message to a RAGAS message object."""
        content = str(self.content) if self.content is not None else ""
        role_lower = self.role.lower()
        
        if role_lower in ("human", "user", "system"):
            # System messages are converted to HumanMessage as RAGAS doesn't have SystemMessage
            return HumanMessage(content=content)
        
        if role_lower in ("assistant", "ai"):
            tool_calls = self._get_tool_calls()
            return AIMessage(content=content, tool_calls=tool_calls)
        
        if role_lower == "tool":
            return ToolMessage(content=content)
        
        # Default to HumanMessage if role is unknown
        return HumanMessage(content=content)


def convert_message_dict_to_ragas_message(msg_dict: Dict[str, Any] | Any) -> HumanMessage | AIMessage | ToolMessage:
    """
    Convert a message dictionary to the appropriate RAGAS message object.
    
    Args:
        msg_dict: Dictionary with 'role' and 'content' fields, or already a message object
        
    Returns:
        RAGAS message object (HumanMessage, AIMessage, or ToolMessage)
    """
    # If it's already a message object, return as-is
    if isinstance(msg_dict, (HumanMessage, AIMessage, ToolMessage)):
        return msg_dict
    
    # If it's not a dict, try to convert it
    if not isinstance(msg_dict, dict):
        return HumanMessage(content=str(msg_dict))
    
    # Use Pydantic model to parse and validate the message
    try:
        message = ConversationMessage(**msg_dict)
        return message.to_ragas_message()
    except Exception:
        # Fallback to basic conversion if Pydantic parsing fails
        role = msg_dict.get("role", "").lower()
        content_raw = msg_dict.get("content", "")
        content = str(content_raw) if content_raw is not None else ""
        
        if role in ("human", "user", "system"):
            return HumanMessage(content=content)
        elif role in ("assistant", "ai"):
            return AIMessage(content=content)
        elif role == "tool":
            return ToolMessage(content=content)
        else:
            return HumanMessage(content=content)


def convert_tool_call_dict_to_ragas_tool_call(tc_dict: Dict[str, Any] | ToolCall) -> ToolCall:
    """
    Convert a tool call dictionary to a RAGAS ToolCall object.
    
    Args:
        tc_dict: Dictionary with 'name' and 'args' fields, or already a ToolCall object
        
    Returns:
        RAGAS ToolCall object
    """
    if isinstance(tc_dict, ToolCall):
        return tc_dict
    
    if not isinstance(tc_dict, dict):
        raise ValueError(f"Cannot convert {type(tc_dict)} to ToolCall")
    
    return ToolCall(
        name=tc_dict.get("name", ""),
        args=tc_dict.get("args", {})
    )


def convert_message_list_to_ragas_messages(
    messages: List[Dict[str, Any] | HumanMessage | AIMessage | ToolMessage]
) -> List[HumanMessage | AIMessage | ToolMessage]:
    """
    Convert a list of message dictionaries to RAGAS message objects.
    
    Args:
        messages: List of message dictionaries or message objects
        
    Returns:
        List of RAGAS message objects
    """
    return [convert_message_dict_to_ragas_message(msg) for msg in messages]


def convert_tool_call_list_to_ragas_tool_calls(
    tool_calls: List[Dict[str, Any] | ToolCall]
) -> List[ToolCall]:
    """
    Convert a list of tool call dictionaries to RAGAS ToolCall objects.
    
    Args:
        tool_calls: List of tool call dictionaries or ToolCall objects
        
    Returns:
        List of RAGAS ToolCall objects
    """
    return [convert_tool_call_dict_to_ragas_tool_call(tc) for tc in tool_calls]

