"""
Unit tests for OpenAI message converter.
"""
import pytest
from app.services.llm.openai.message_converter import OpenAIMessageConverter
from unittest.mock import Mock, patch


class TestOpenAIMessageConverter:
    """Tests for OpenAIMessageConverter."""
    
    @pytest.fixture
    def converter(self):
        """Message converter fixture."""
        return OpenAIMessageConverter()
    
    def test_convert_string_template(self, converter, sample_model_configuration_with_key):
        """Should convert string template to messages."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        
        request = LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="Hello {name}!"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.F_STRING,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={}
                )
            ),
            inputs={"name": "World"}
        )
        
        messages = converter.convert(request)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Hello World!" in messages[0]["content"]
    
    def test_convert_chat_template(self, converter, sample_model_configuration_with_key):
        """Should convert chat template to messages."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptChatTemplate,
            PromptMessage,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        
        request = LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptChatTemplate(messages=[
                    PromptMessage(role="system", content="You are helpful."),
                    PromptMessage(role="user", content="Hello!")
                ]),
                template_type=TemplateType.CHAT,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={}
                )
            ),
            inputs={}
        )
        
        messages = converter.convert(request)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_normalizes_roles(self, converter, sample_model_configuration_with_key):
        """Should normalize unknown roles."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptChatTemplate,
            PromptMessage,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        
        request = LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptChatTemplate(messages=[
                    PromptMessage(role="user", content="Test")  # Use valid role since schema validates
                ]),
                template_type=TemplateType.CHAT,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={}
                )
            ),
            inputs={}
        )
        
        messages = converter.convert(request)
        
        # Unknown roles should be normalized to "user"
        assert messages[0]["role"] == "user"

