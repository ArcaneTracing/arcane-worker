"""
Anthropic (Claude) model service implementation.

Handles LLM requests for Anthropic Claude models.
Supports system messages and Anthropic-specific message format.
"""
from typing import Dict, Any, List, Optional
from app.models.schemas import (
    LLMServiceRequestDto,
    AdapterType,
    TokenUsage,
    ModelInfo,
    ToolCall,
)
from app.services.llm.base import BaseModelService
from app.services.llm.clients import create_anthropic_client
from app.services.template import TemplateService
import asyncio


class AnthropicModelService(BaseModelService):
    """Service for Anthropic (Claude) models"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.client = create_anthropic_client(api_key=api_key, base_url=base_url, timeout=timeout)
        self.template_service = TemplateService()
    
    def get_adapter_type(self) -> str:
        return AdapterType.ANTHROPIC.value
    
    def _render_template(self, request: LLMServiceRequestDto):
        """Render template using template service"""
        return self.template_service.render_template(
            template=request.prompt_version.template,
            template_format=request.prompt_version.template_format,
            inputs=request.inputs
        )
    
    def _convert_messages(self, rendered_template) -> tuple[str | None, List[Dict[str, str]]]:
        """Convert rendered template to Anthropic format (separates system message)"""
        from app.core.message_utils import extract_text_from_content, normalize_role
        
        system_message = None
        conversation_messages = []
        
        if rendered_template.type == "chat":
            for msg in rendered_template.messages:
                content = extract_text_from_content(msg.content)
                
                # Anthropic separates system messages
                if msg.role == "system":
                    system_message = content
                else:
                    # Map roles - Anthropic uses user and assistant
                    role = normalize_role(msg.role, supported_roles={"user", "assistant"})
                    
                    conversation_messages.append({
                        "role": role,
                        "content": content
                    })
        elif rendered_template.type == "string":
            conversation_messages.append({
                "role": "user",
                "content": rendered_template.template
            })
        
        # Anthropic requires at least one message (e.g. evaluations with only system message)
        if not conversation_messages:
            conversation_messages.append({"role": "user", "content": " "})
        
        return system_message, conversation_messages
    
    def _extract_parameters(self, request: LLMServiceRequestDto) -> tuple[float, int]:
        """Extract temperature and max_tokens from invocation parameters or model config"""
        from app.core.parameter_extractor import extract_temperature_and_max_tokens_anthropic
        return extract_temperature_and_max_tokens_anthropic(request)
    
    def _get_tools(self, request: LLMServiceRequestDto) -> List[Dict[str, Any]] | None:
        """Pass-through tools. User provides Anthropic format; errors surface from API."""
        if not (request.prompt_version.tools and request.prompt_version.tools.tools):
            return None
        return request.prompt_version.tools.tools
    
    def _build_request_params(
        self,
        model_name: str,
        max_tokens: int,
        temperature: float,
        conversation_messages: List[Dict[str, str]],
        system_message: str | None,
        tools_dict: List[Dict[str, Any]] | None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build request parameters for Anthropic API."""
        params = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation_messages
        }
        
        if system_message:
            params["system"] = system_message
        
        if tools_dict:
            params["tools"] = tools_dict

        # Structured output: wrap response_format in output_config.format
        if response_format:
            params["output_config"] = {"format": response_format}
        
        return params
    
    def _extract_output(self, response) -> str:
        """Extract output text from response."""
        return response.content[0].text if response.content else ""
    
    def _extract_tool_calls(self, response) -> List[Dict[str, Any]] | None:
        """Extract tool calls from response if present."""
        if not (hasattr(response, 'content') and response.content):
            return None
        
        tool_calls = []
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                tool_calls.append(ToolCall(
                    id=content_block.id,
                    name=content_block.name,
                    arguments=content_block.input
                ).dict())
        
        return tool_calls if tool_calls else None
    
    def _extract_usage(self, response) -> Dict[str, Any] | None:
        """Extract usage information from response if present."""
        if not (hasattr(response, 'usage') and response.usage):
            return None
        
        return TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        ).dict()
    
    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """Execute Anthropic request with full request DTO"""
        # Render template
        rendered_template = self._render_template(request)
        
        # Convert to Anthropic message format (separates system message)
        system_message, conversation_messages = self._convert_messages(rendered_template)
        
        # Extract parameters
        temperature, max_tokens = self._extract_parameters(request)
        
        # Get model name
        model_name = request.model_configuration.configuration.model_name
        
        # Pass-through tools (user provides provider-specific format)
        tools_dict = self._get_tools(request)

        # Build request parameters
        params = self._build_request_params(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            conversation_messages=conversation_messages,
            system_message=system_message,
            tools_dict=tools_dict,
            response_format=request.prompt_version.response_format,
        )
        
        # Execute API call
        response = await asyncio.to_thread(
            lambda: self.client.messages.create(**params)
        )
        
        # Extract response data
        output = self._extract_output(response)
        finish_reason = response.stop_reason
        tool_calls = self._extract_tool_calls(response)
        usage = self._extract_usage(response)
        model_info = ModelInfo(id=model_name, name=model_name).dict()
        
        return {
            "output": output,
            "usage": usage,
            "model": model_info,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls
        }

