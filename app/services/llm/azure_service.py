"""
Azure OpenAI model service implementation.

Handles LLM requests for Azure OpenAI models.
Uses Azure-specific endpoint and deployment configuration.
"""
from typing import Dict, Any, List
from app.models.schemas import (
    LLMServiceRequestDto,
    AdapterType,
    TokenUsage,
    ModelInfo,
    ToolCall,
)
from app.services.llm.base import BaseModelService
from app.services.llm.clients import create_azure_openai_client
from app.services.template import TemplateService
import asyncio
import json


class AzureModelService(BaseModelService):
    """Service for Azure OpenAI models"""
    
    def __init__(self, api_key: str, endpoint: str, api_version: str = "2024-02-15-preview"):
        self.client = create_azure_openai_client(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            use_azure_class=True
        )
        self.template_service = TemplateService()
    
    def get_adapter_type(self) -> str:
        return AdapterType.AZURE.value
    
    def _render_template(self, request: LLMServiceRequestDto):
        """Render template using template service"""
        return self.template_service.render_template(
            template=request.prompt_version.template,
            template_format=request.prompt_version.template_format,
            inputs=request.inputs
        )
    
    def _convert_messages(self, rendered_template) -> List[Dict[str, str]]:
        """Convert rendered template to Azure OpenAI message format (same as OpenAI)"""
        from app.core.message_utils import extract_text_from_content, normalize_role
        
        messages = []
        if rendered_template.type == "chat":
            for msg in rendered_template.messages:
                content = extract_text_from_content(msg.content)
                role = normalize_role(msg.role)
                
                messages.append({
                    "role": role,
                    "content": content
                })
        elif rendered_template.type == "string":
            messages.append({
                "role": "user",
                "content": rendered_template.template
            })
        
        return messages
    
    def _extract_parameters(self, request: LLMServiceRequestDto) -> tuple[float, int]:
        """Extract temperature and max_tokens from invocation parameters or model config"""
        from app.core.parameter_extractor import extract_temperature_and_max_tokens_openai
        return extract_temperature_and_max_tokens_openai(request)
    
    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """Execute Azure OpenAI request with full request DTO"""
        # Render template
        rendered_template = self._render_template(request)
        
        # Convert to messages format
        messages = self._convert_messages(rendered_template)
        
        # Extract parameters
        temperature, max_tokens = self._extract_parameters(request)
        
        # Get model name
        model_name = request.model_configuration.configuration.model_name
        
        # Prepare tools if provided
        from app.core.message_utils import convert_tools_to_format
        tools_dict = convert_tools_to_format(request.prompt_version.tools, "openai")
        
        # Pass-through response format (user provides provider-specific format)
        response_format_dict = request.prompt_version.response_format
        
        def _create_completion():
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools_dict:
                params["tools"] = tools_dict
            if response_format_dict:
                params["response_format"] = response_format_dict
            
            return self.client.chat.completions.create(**params)
        
        response = await asyncio.to_thread(_create_completion)
        
        output = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        
        # Extract tool calls if present
        tool_calls = None
        if response.choices[0].message.tool_calls:
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
        
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ).dict()
        
        model_info = ModelInfo(id=model_name, name=model_name).dict()
        
        return {
            "output": output,
            "usage": usage,
            "model": model_info,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls
        }

