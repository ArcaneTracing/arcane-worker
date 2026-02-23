"""
OpenAI parameter extraction utilities.

Extracts and converts parameters from request DTOs to OpenAI API format.
Handles temperature, max_tokens, tools, and response format.
"""
from typing import List, Dict, Any, Optional
from app.models.schemas import LLMServiceRequestDto
from app.core.parameter_extractor import extract_temperature_and_max_tokens_openai
from app.core.message_utils import convert_tools_to_format


class OpenAIParameterExtractor:
    """Extracts parameters from request for OpenAI API"""
    
    def extract_temperature_and_max_tokens(self, request: LLMServiceRequestDto) -> tuple[float, int]:
        """Extract temperature and max_tokens from invocation parameters or model config"""
        return extract_temperature_and_max_tokens_openai(request)
    
    def extract_tools(self, request: LLMServiceRequestDto) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to OpenAI format"""
        return convert_tools_to_format(request.prompt_version.tools, "openai")
    
    def extract_response_format(self, request: LLMServiceRequestDto) -> Optional[Dict[str, Any]]:
        """Pass-through response format. User provides provider-specific format."""
        return request.prompt_version.response_format

