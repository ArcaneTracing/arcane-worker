"""
Unit tests for parameter extractors.
"""
import pytest
from app.core.parameter_extractor import (
    extract_temperature_and_max_tokens_openai,
    extract_temperature_and_max_tokens_anthropic
)
from unittest.mock import Mock


class TestParameterExtractor:
    """Tests for parameter extractor functions."""
    
    def test_extract_openai_params_from_invocation(self, sample_model_configuration_with_key):
        """Should extract params from OpenAI invocation parameters."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        # Update model config to have temperature/max_tokens
        sample_model_configuration_with_key.configuration.temperature = 0.5
        sample_model_configuration_with_key.configuration.max_tokens = 500
        
        request = LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={
                        "temperature": 0.8,
                        "max_tokens": 1000
                    }
                )
            ),
            inputs={}
        )
        
        temp, max_tokens = extract_temperature_and_max_tokens_openai(request)
        assert temp == 0.8  # From invocation params
        assert max_tokens == 1000  # From invocation params
    
    def test_extract_openai_params_from_config(self, sample_model_configuration_with_key):
        """Should extract params from model config if not in invocation."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat,
            OpenAIInvocationParameters
        )
        # Update model config to have temperature/max_tokens
        sample_model_configuration_with_key.configuration.temperature = 0.7
        sample_model_configuration_with_key.configuration.max_tokens = 2000
        
        request = LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={}  # Empty, should use config defaults
                )
            ),
            inputs={}
        )
        
        temp, max_tokens = extract_temperature_and_max_tokens_openai(request)
        assert temp == 0.7  # From config
        assert max_tokens == 2000  # From config
    
    def test_extract_openai_params_defaults(self, sample_model_configuration_with_key):
        """Should use defaults if params not provided."""
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
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=OpenAIInvocationParameters(
                    type="openai",
                    openai={}  # Empty, should use defaults
                )
            ),
            inputs={}
        )
        
        temp, max_tokens = extract_temperature_and_max_tokens_openai(request)
        assert temp == 0.7  # Default
        assert max_tokens == 1000  # Default
    
    def test_extract_anthropic_params_from_invocation(self, sample_model_configuration_with_key):
        """Should extract params from Anthropic invocation parameters."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            BaseModelConfiguration,
            AdapterType,
            ModelConfigurationWithEncryptedKey,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat,
            AnthropicInvocationParameters
        )
        # Create Anthropic config
        anthropic_config = ModelConfigurationWithEncryptedKey(
            id="config-1",
            name="Anthropic Config",
            configuration=BaseModelConfiguration(
                adapter=AdapterType.ANTHROPIC,
                model_name="claude-3",
                api_key="test-key",
                temperature=0.5,
                max_tokens=500
            ),
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z"
        )
        
        request = LLMServiceRequestDto(
            model_configuration=anthropic_config,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=AnthropicInvocationParameters(
                    type="anthropic",
                    anthropic={
                        "temperature": 0.9,
                        "max_tokens": 1500
                    }
                )
            ),
            inputs={}
        )
        
        temp, max_tokens = extract_temperature_and_max_tokens_anthropic(request)
        assert temp == 0.9  # From invocation params
        assert max_tokens == 1500  # From invocation params

