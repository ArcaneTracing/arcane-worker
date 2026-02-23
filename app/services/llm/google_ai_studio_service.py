"""
Google AI Studio (Gemini API) model service implementation.

Uses the Google GenAI SDK (google-genai) per the migration guide:
https://ai.google.dev/gemini-api/docs/migrate

Handles LLM requests for Gemini models via the Developer API.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from app.models.schemas import (
    LLMServiceRequestDto,
    AdapterType,
    ModelInfo,
)
from app.services.llm.base import BaseModelService
from app.services.template import TemplateService
from app.core.message_utils import extract_text_from_content, normalize_role
from app.core.parameter_extractor import extract_temperature_and_max_tokens_google_ai_studio
from app.core.google_response_utils import extract_output_and_tool_calls

class GoogleAIStudioModelService(BaseModelService):
    """Service for Google AI Studio (Gemini API) models."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.template_service = TemplateService()

    def get_adapter_type(self) -> str:
        return AdapterType.GOOGLE_AI_STUDIO.value

    def _render_template(self, request: LLMServiceRequestDto):
        """Render template using template service."""
        return self.template_service.render_template(
            template=request.prompt_version.template,
            template_format=request.prompt_version.template_format,
            inputs=request.inputs,
        )

    def _convert_messages(
        self, rendered_template
    ) -> tuple[str | None, List[Dict[str, Any]]]:
        """
        Convert rendered template to Gemini format.
        Gemini uses 'user' and 'model' roles.

        Returns:
            (system_instruction, contents) - system_instruction for system message,
            contents as list of Content dicts for generate_content.
        """
        system_instruction = None
        contents: list[Dict[str, Any]] = []

        if rendered_template.type == "chat":
            for msg in rendered_template.messages:
                content = extract_text_from_content(msg.content)
                # Gemini uses "user" and "model" (not "assistant")
                role = normalize_role(
                    msg.role, supported_roles={"user", "assistant", "model"}
                )
                if role == "assistant":
                    role = "model"

                if msg.role == "system":
                    system_instruction = content
                else:
                    contents.append(
                        {
                            "role": role,
                            "parts": [{"text": content}],
                        }
                    )
        elif rendered_template.type == "string":
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": rendered_template.template}],
                }
            )

        # Gemini requires at least one content (e.g. evaluations with only system message)
        if not contents:
            contents.append({"role": "user", "parts": [{"text": " "}]})

        return system_instruction, contents

    def _extract_parameters(self, request: LLMServiceRequestDto) -> tuple[float, int]:
        """Extract temperature and max_tokens from invocation parameters or config."""
        return extract_temperature_and_max_tokens_google_ai_studio(request)

    def _extract_top_p(self, request: LLMServiceRequestDto) -> Optional[float]:
        """Extract top_p from invocation parameters if present."""
        invocation_params = request.prompt_version.invocation_parameters
        params = None
        if invocation_params.type == "google-ai-studio":
            params = getattr(invocation_params, "google_ai_studio", None)
        elif invocation_params.type == "google":
            params = getattr(invocation_params, "google", None)
        if isinstance(params, dict):
            return params.get("top_p") or params.get("topP")
        return request.model_configuration.configuration.top_p


    def _extract_usage(self, response) -> Optional[Dict[str, Any]]:
        """Extract usage information from response if present."""
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            return None
        um = response.usage_metadata
        prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
        completion_tokens = getattr(um, "candidates_token_count", 0) or 0
        total = getattr(um, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        return {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total,
        }

    def _extract_finish_reason(self, response) -> Optional[str]:
        """Extract finish reason from response."""
        if not response or not response.candidates:
            return None
        return getattr(response.candidates[0], "finish_reason", None)

    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """Execute Google AI Studio (Gemini) request with full request DTO."""
        rendered_template = self._render_template(request)
        system_instruction, contents = self._convert_messages(rendered_template)
        temperature, max_tokens = self._extract_parameters(request)
        top_p = self._extract_top_p(request)

        model_name = request.model_configuration.configuration.model_name

        config_dict: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if top_p is not None:
            config_dict["top_p"] = top_p
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        # Pass-through response format (user provides provider-specific format, e.g. responseMimeType + responseSchema)
        if request.prompt_version.response_format:
            for key, value in request.prompt_version.response_format.items():
                if value is not None:
                    config_dict[key] = value

        # Pass-through tools (user provides provider-specific format, e.g. [{ functionDeclarations: [...] }])
        if request.prompt_version.tools and request.prompt_version.tools.tools:
            config_dict["tools"] = request.prompt_version.tools.tools

        config = self._types.GenerateContentConfig(**config_dict)

        # For single user message, pass string; otherwise pass list of contents
        if len(contents) == 1 and contents[0]["role"] == "user":
            contents_arg = contents[0]["parts"][0]["text"]
        else:
            contents_arg = contents

        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=contents_arg,
            config=config,
        )

        output, tool_calls = extract_output_and_tool_calls(response)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        model_info = ModelInfo(id=model_name, name=model_name).model_dump()

        return {
            "output": output,
            "usage": usage,
            "model": model_info,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }
