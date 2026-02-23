"""
Unit tests for Anthropic service.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.llm.anthropic_service import AnthropicModelService
from app.models.schemas import (
    LLMServiceRequestDto,
    ModelConfigurationWithEncryptedKey,
    BaseModelConfiguration,
    AdapterType,
    PromptVersionDto,
    PromptStringTemplate,
    TemplateType,
    TemplateFormat,
    AnthropicInvocationParameters
)


class TestAnthropicService:
    """Tests for AnthropicModelService."""
    
    @pytest.fixture
    def service(self):
        """Service fixture."""
        return AnthropicModelService(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_get_adapter_type(self, service):
        """Should return Anthropic adapter type."""
        assert service.get_adapter_type() == AdapterType.ANTHROPIC.value

