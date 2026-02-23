"""
OpenAI API response processor.

Processes OpenAI chat completion responses into standardized format.
Extracts output text, token usage, tool calls, and metadata.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

from app.models.schemas import TokenUsage, ModelInfo, ToolCall
import json


class OpenAIResponseProcessor:
    """Processes OpenAI API responses into standardized format"""
    
    def process(self, response: ChatCompletion, model_name: str) -> Dict[str, Any]:
        """Process OpenAI response into standardized format"""
        message = response.choices[0].message
        
        return {
            "output": message.content or "",
            "usage": self._extract_usage(response),
            "model": ModelInfo(id=model_name, name=model_name).dict(),
            "finish_reason": response.choices[0].finish_reason,
            "tool_calls": self._extract_tool_calls(response)
        }
    
    def _extract_tool_calls(self, response: ChatCompletion) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from OpenAI response"""
        if not response.choices[0].message.tool_calls:
            return None
        
        tool_calls = []
        from app.core.security import safe_json_loads
        
        for tc in response.choices[0].message.tool_calls:
            # Parse JSON arguments if string (with security validation)
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                arguments = safe_json_loads(arguments, fallback={})
            
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments
                ).dict()
            )
        
        return tool_calls
    
    def _extract_usage(self, response: ChatCompletion) -> Optional[Dict[str, Any]]:
        """Extract token usage from OpenAI response"""
        if not hasattr(response, 'usage') or not response.usage:
            return None
        
        return TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        ).dict()

