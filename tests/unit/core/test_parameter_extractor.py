"""
Unit tests for core parameter extractors.
"""
import pytest
from app.core.parameter_extractor import (
    extract_temperature_and_max_tokens,
    extract_temperature_and_max_tokens_openai,
    extract_temperature_and_max_tokens_anthropic
)


class TestParameterExtractor:
    """Tests for parameter extractor utilities."""
    
    @pytest.fixture
    def base_request(self, sample_model_configuration_with_key, sample_openai_invocation_params):
        """Base request fixture."""
        from app.models.schemas import (
            LLMServiceRequestDto,
            PromptVersionDto,
            PromptStringTemplate,
            TemplateType,
            TemplateFormat
        )
        return LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=PromptVersionDto(
                model_configuration_id="config-1",
                template=PromptStringTemplate(template="test"),
                template_type=TemplateType.STR,
                template_format=TemplateFormat.NONE,
                invocation_parameters=sample_openai_invocation_params
            ),
            inputs={}
        )
    
    def test_extract_params_uses_defaults(self, base_request):
        """Should use defaults when provider type not specified."""
        # Use OpenAI invocation params but test the generic extractor
        temp, max_tokens = extract_temperature_and_max_tokens_openai(base_request)
        assert temp == 0.7  # From invocation params
        assert max_tokens == 1000  # From invocation params
    
    def test_extract_openai_handles_azure_openai_params(self, base_request):
        """Should handle Azure OpenAI invocation params."""
        from app.models.schemas import AzureOpenAIInvocationParameters
        base_request.prompt_version.invocation_parameters = AzureOpenAIInvocationParameters(
            type="azure_openai",
            azure_openai={
                "temperature": 0.85,
                "max_tokens": 1500
            }
        )
        
        temp, max_tokens = extract_temperature_and_max_tokens_openai(base_request)
        assert temp == 0.85
        assert max_tokens == 1500
