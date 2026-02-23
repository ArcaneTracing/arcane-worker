"""
OpenAI model service implementation.

Handles LLM requests for OpenAI models using the OpenAI API.
Manages template rendering, message conversion, and response processing.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from app.models.schemas import LLMServiceRequestDto, AdapterType
from app.services.llm.base import BaseModelService
from app.services.llm.openai.message_converter import OpenAIMessageConverter
from app.services.llm.openai.request_builder import OpenAIRequestBuilder
from app.services.llm.openai.response_processor import OpenAIResponseProcessor
from app.services.llm.clients import create_openai_client
import asyncio
import json

class OpenAIModelService(BaseModelService):
    """Service for OpenAI models"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None
    ) -> None:
        self.client = create_openai_client(api_key=api_key, base_url=base_url, organization=organization)
        self.message_converter = OpenAIMessageConverter()
        self.request_builder = OpenAIRequestBuilder()
        self.response_processor = OpenAIResponseProcessor()
    
    def get_adapter_type(self) -> str:
        return AdapterType.OPENAI.value
    
    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """Execute OpenAI request with full request DTO"""
        try:
            # Convert template to messages
            messages = self.message_converter.convert(request)
            
            # Build API request parameters
            params = self.request_builder.build(request, messages)
            # print the params as json
            print(json.dumps(params, indent=4))
            
            # Execute API call
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(**params)
            )
            
            # Process and return response
            model_name = request.model_configuration.configuration.model_name
            return self.response_processor.process(response, model_name)
        
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="OpenAIModelService",
                context="execute()",
                additional_info={"model": request.model_configuration.configuration.model_name}
            )
            raise

