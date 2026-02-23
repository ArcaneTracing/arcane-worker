"""
Unit tests for OpenAI request builder.
"""
import pytest
from app.services.llm.openai.request_builder import OpenAIRequestBuilder
from app.models.schemas import (
    LLMServiceRequestDto,
    ModelConfigurationWithEncryptedKey,
    BaseModelConfiguration,
    AdapterType,
    PromptVersionDto,
    PromptStringTemplate,
    TemplateType,
    TemplateFormat,
    OpenAIInvocationParameters
)
from unittest.mock import Mock, patch


class TestOpenAIRequestBuilder:
    """Tests for OpenAIRequestBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Request builder fixture."""
        return OpenAIRequestBuilder()
    
    @pytest.fixture
    def sample_request(self, sample_model_configuration_with_key):
        """Sample LLM request."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        
        return LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={"temperature": 0.7, "max_tokens": 1000}
                )
            ),
            inputs={}
        )
    
    def test_build_basic_request(self, builder, sample_request):
        """Should build basic request parameters."""
        messages = [{"role": "user", "content": "test"}]
        
        params = builder.build(sample_request, messages)
        
        assert params["model"] == "gpt-4"
        assert params["messages"] == messages
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
    
    def test_build_with_tools(self, builder, sample_request):
        """Should pass-through tools if provided."""
        from app.models.schemas import Tools
        
        sample_request.prompt_version.tools = Tools(
            type="tools",
            tools=[{"type": "function", "function": {"name": "test_func"}}]
        )
        
        messages = [{"role": "user", "content": "test"}]
        params = builder.build(sample_request, messages)
        
        assert "tools" in params
        assert len(params["tools"]) == 1
        assert params["tools"][0]["type"] == "function"
    
    def test_build_without_tools(self, builder, sample_request):
        """Should not include tools if not provided."""
        messages = [{"role": "user", "content": "test"}]
        params = builder.build(sample_request, messages)
        
        assert "tools" not in params
    
    def test_build_with_response_format(self, builder, sample_request):
        """Should pass-through response format if provided."""
        sample_request.prompt_version.response_format = {
            "type": "json_schema",
            "json_schema": {"type": "object", "properties": {}}
        }
        
        messages = [{"role": "user", "content": "test"}]
        params = builder.build(sample_request, messages)
        
        assert "response_format" in params
        assert params["response_format"]["type"] == "json_schema"

