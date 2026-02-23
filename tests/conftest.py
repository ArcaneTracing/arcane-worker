"""
Pytest configuration and shared fixtures.
"""
import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock
from app.models.schemas import (
    BaseModelConfiguration,
    ModelConfigurationWithEncryptedKey,
    AdapterType,
    PromptVersionDto,
    PromptStringTemplate,
    PromptChatTemplate,
    TemplateType,
    TemplateFormat,
    OpenAIInvocationParameters,
    AzureOpenAIInvocationParameters,
    AnthropicInvocationParameters,
    LLMServiceRequestDto
)


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration for testing."""
    return {
        "configuration": {
            "adapter": "openai",
            "modelName": "gpt-4",
            "apiKey": "test-api-key",
            "temperature": 0.7,
            "maxTokens": 1000,
            "config": {}
        }
    }


@pytest.fixture
def sample_ragas_config() -> Dict[str, Any]:
    """Sample RAGAS model configuration for testing."""
    return {
        "configuration": {
            "adapter": "openai",
            "modelName": "gpt-4",
            "apiKey": "test-api-key",
            "config": {}
        }
    }


@pytest.fixture
def sample_score_mapping() -> Dict[str, Any]:
    """Sample score mapping for RAGAS evaluation."""
    return {
        "user_input": "What is the capital of France?",
        "contexts": ["Paris is the capital of France."],
        "response": "The capital of France is Paris."
    }


@pytest.fixture
def sample_model_configuration_with_key():
    """Sample model config with decrypted api_key (as received from backend for LLM calls)."""
    return ModelConfigurationWithEncryptedKey(
        id="config-1",
        name="Test Config",
        configuration=BaseModelConfiguration(
            adapter=AdapterType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        ),
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z"
    )


@pytest.fixture
def sample_openai_invocation_params():
    """Sample OpenAI invocation parameters."""
    return OpenAIInvocationParameters(
        type="openai",
        openai={"temperature": 0.7, "max_tokens": 1000}
    )


@pytest.fixture
def sample_azure_invocation_params():
    """Sample Azure OpenAI invocation parameters."""
    return AzureOpenAIInvocationParameters(
        type="azure_openai",
        azure_openai={"temperature": 0.7, "max_tokens": 1000}
    )


@pytest.fixture
def sample_anthropic_invocation_params():
    """Sample Anthropic invocation parameters."""
    return AnthropicInvocationParameters(
        type="anthropic",
        anthropic={"temperature": 0.7, "max_tokens": 1000}
    )


@pytest.fixture
def sample_prompt_version(sample_openai_invocation_params):
    """Sample prompt version with required fields."""
    return PromptVersionDto(
        model_configuration_id="config-1",
        template=PromptStringTemplate(template="test"),
        template_type=TemplateType.STR,
        template_format=TemplateFormat.NONE,
        invocation_parameters=sample_openai_invocation_params
    )


@pytest.fixture
def sample_llm_request(sample_model_configuration_with_key, sample_prompt_version):
    """Sample LLM service request for testing."""
    return LLMServiceRequestDto(
        model_configuration=sample_model_configuration_with_key,
        prompt_version=sample_prompt_version,
        inputs={"question": "What is the capital of France?"}
    )


@pytest.fixture
def mock_llm():
    """Mock LLM instance for testing."""
    mock = AsyncMock()
    mock.__class__.__name__ = "MockLLM"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings instance for testing."""
    mock = AsyncMock()
    mock.__class__.__name__ = "MockEmbeddings"
    return mock


@pytest.fixture
def mock_ragas_score():
    """Mock RAGAS score for testing."""
    mock = Mock()
    mock.score = 0.85
    mock.metric = "context_precision"
    mock.id = "test-metric-id"
    return mock


@pytest.fixture
def mock_model_config_client():
    """Mock model config client for testing."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_llm_builder():
    """Mock LLM builder for testing."""
    mock = Mock()
    mock.build_from_config = Mock(return_value=Mock())
    return mock


@pytest.fixture
def mock_embeddings_builder():
    """Mock embeddings builder for testing."""
    mock = Mock()
    mock.build_from_config = Mock(return_value=Mock())
    return mock


@pytest.fixture(autouse=True)
def reset_registries():
    """
    Reset registries before each test to ensure isolation.
    
    Note: This requires access to internal registry state which may need
    to be implemented based on the actual registry implementation.
    """
    yield
    # Cleanup after test if needed
