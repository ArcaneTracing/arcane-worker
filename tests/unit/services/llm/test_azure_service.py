"""
Unit tests for Azure OpenAI service.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.llm.azure_service import AzureModelService
from app.models.schemas import AdapterType


class TestAzureOpenAIService:
    """Tests for AzureModelService."""
    
    @pytest.fixture
    def service(self):
        """Service fixture."""
        return AzureModelService(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview"
        )
    
    @pytest.mark.asyncio
    async def test_get_adapter_type(self, service):
        """Should return Azure adapter type."""
        assert service.get_adapter_type() == AdapterType.AZURE.value

