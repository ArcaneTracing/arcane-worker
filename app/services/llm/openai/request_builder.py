"""
OpenAI API request builder.

Builds complete request parameters for OpenAI chat completions API.
Handles model name, messages, temperature, max_tokens, tools, and response format.
"""
from __future__ import annotations

from typing import List, Dict, Any
from app.models.schemas import LLMServiceRequestDto
from app.services.llm.openai.parameter_extractor import OpenAIParameterExtractor


class OpenAIRequestBuilder:
    """Builds OpenAI API request parameters"""
    
    def __init__(self) -> None:
        self.parameter_extractor = OpenAIParameterExtractor()
    
    def build(
        self,
        request: LLMServiceRequestDto,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Build complete request parameters for OpenAI API"""
        model_name = request.model_configuration.configuration.model_name
        temperature, max_tokens = self.parameter_extractor.extract_temperature_and_max_tokens(request)
        tools = self.parameter_extractor.extract_tools(request)
        response_format = self.parameter_extractor.extract_response_format(request)
        
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if tools:
            params["tools"] = tools
        if response_format:
            params["response_format"] = response_format
        
        return params

