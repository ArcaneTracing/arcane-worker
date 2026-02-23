"""
AWS Bedrock model service implementation.

Uses the Bedrock Converse API via boto3.
https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_Converse_AmazonNovaText_section.html

Handles LLM requests for Amazon Nova and other Bedrock models.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional

from app.models.schemas import (
    LLMServiceRequestDto,
    AdapterType,
    ModelInfo,
)
from app.services.llm.base import BaseModelService
from app.services.template import TemplateService
from app.core.message_utils import extract_text_from_content, normalize_role
from app.core.parameter_extractor import extract_temperature_and_max_tokens_bedrock

logger = logging.getLogger(__name__)


class BedrockModelService(BaseModelService):
    """Service for AWS Bedrock models using Converse API."""

    def __init__(
        self,
        region: str,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        import boto3

        client_kwargs: Dict[str, Any] = {
            "service_name": "bedrock-runtime",
            "region_name": region,
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        # Config with reasonable timeout for LLM inference
        from botocore.config import Config
        config = Config(read_timeout=300, connect_timeout=10)
        client_kwargs["config"] = config

        self.client = boto3.client(**client_kwargs)
        self.template_service = TemplateService()

    def get_adapter_type(self) -> str:
        return AdapterType.BEDROCK.value

    def _render_template(self, request: LLMServiceRequestDto):
        """Render template using template service."""
        return self.template_service.render_template(
            template=request.prompt_version.template,
            template_format=request.prompt_version.template_format,
            inputs=request.inputs,
        )

    def _convert_messages(
        self, rendered_template
    ) -> tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Convert rendered template to Bedrock Converse format.
        messages: [{"role": "user"|"assistant", "content": [{"text": "..."}]}]
        system: [{"text": "..."}] for system message
        """
        system_blocks: Optional[List[Dict[str, Any]]] = None
        messages: List[Dict[str, Any]] = []

        if rendered_template.type == "chat":
            for msg in rendered_template.messages:
                content = extract_text_from_content(msg.content)
                role = normalize_role(
                    msg.role, supported_roles={"user", "assistant"}
                )

                if msg.role == "system":
                    system_blocks = [{"text": content}]
                else:
                    messages.append({
                        "role": role,
                        "content": [{"text": content}],
                    })
        elif rendered_template.type == "string":
            messages.append({
                "role": "user",
                "content": [{"text": rendered_template.template}],
            })

        return system_blocks, messages

    def _extract_parameters(self, request: LLMServiceRequestDto) -> tuple[float, int]:
        """Extract temperature and max_tokens from invocation parameters or config."""
        return extract_temperature_and_max_tokens_bedrock(request)

    def _extract_top_p(self, request: LLMServiceRequestDto) -> Optional[float]:
        """Extract top_p from invocation parameters if present."""
        invocation_params = request.prompt_version.invocation_parameters
        params = None
        if hasattr(invocation_params, "type"):
            if invocation_params.type == "bedrock":
                params = getattr(invocation_params, "bedrock", None)
            elif invocation_params.type == "aws":
                params = getattr(invocation_params, "aws", None)
        if isinstance(params, dict):
            return params.get("top_p") or params.get("topP")
        return request.model_configuration.configuration.top_p

    def _extract_output(self, response: Dict[str, Any]) -> str:
        """Extract output text from Converse API response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        if content and isinstance(content[0], dict):
            return content[0].get("text", "") or ""
        return ""

    def _extract_usage(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract usage information from response if present."""
        usage = response.get("usage")
        if not usage:
            return None
        return {
            "promptTokens": usage.get("inputTokens", 0),
            "completionTokens": usage.get("outputTokens", 0),
            "totalTokens": usage.get("totalTokens", 0)
            or usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
        }

    def _extract_finish_reason(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract stop reason from response."""
        output = response.get("output", {})
        return output.get("stopReason")

    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """Execute Bedrock Converse request."""
        rendered_template = self._render_template(request)
        system_blocks, messages = self._convert_messages(rendered_template)
        temperature, max_tokens = self._extract_parameters(request)
        top_p = self._extract_top_p(request)

        model_name = request.model_configuration.configuration.model_name

        inference_config: Dict[str, Any] = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            inference_config["topP"] = top_p

        converse_args: Dict[str, Any] = {
            "modelId": model_name,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system_blocks:
            converse_args["system"] = system_blocks

        response = await asyncio.to_thread(
            lambda: self.client.converse(**converse_args)
        )

        output = self._extract_output(response)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        model_info = ModelInfo(id=model_name, name=model_name).model_dump()

        return {
            "output": output,
            "usage": usage,
            "model": model_info,
            "finish_reason": finish_reason,
            "tool_calls": None,
        }
