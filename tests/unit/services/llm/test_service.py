"""
Unit tests for ModelService.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.services.llm.service import ModelService, get_model_service
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


class TestModelService:
    """Tests for ModelService."""
    
    @pytest.fixture
    def service(self):
        """Service fixture."""
        return ModelService()
    
    @pytest.fixture
    def sample_request(self, sample_model_configuration_with_key, sample_prompt_version):
        """Sample LLM request."""
        return LLMServiceRequestDto(
            model_configuration=sample_model_configuration_with_key,
            prompt_version=sample_prompt_version,
            inputs={}
        )
    
    @pytest.mark.asyncio
    async def test_execute_creates_service_and_runs(self, service, sample_request):
        """Should create service and execute request."""
        from unittest.mock import patch
        mock_service = AsyncMock()
        mock_service.execute = AsyncMock(return_value={"output": "test response"})
        
        with patch('app.services.llm.service.ModelServiceFactory.create_service') as mock_factory:
            mock_factory.return_value = mock_service
            
            result = await service.execute(sample_request)
            
            assert result == {"output": "test response"}
            mock_service.execute.assert_called_once_with(sample_request)
    
    @pytest.mark.asyncio
    async def test_execute_passes_decrypted_api_key_to_factory(self, service, sample_request):
        """Should pass decrypted (plain) api_key from request to factory for LLM calls."""
        from unittest.mock import patch
        mock_service = AsyncMock()
        mock_service.execute = AsyncMock(return_value={"output": "ok"})
        
        with patch('app.services.llm.service.ModelServiceFactory.create_service') as mock_factory:
            mock_factory.return_value = mock_service
            
            await service.execute(sample_request)
            
            call_args = mock_factory.call_args
            api_key = call_args[0][1]
            assert api_key == "test-key"
    
    @pytest.mark.asyncio
    async def test_execute_handles_errors(self, service, sample_request):
        """Should handle errors during execution."""
        from unittest.mock import patch
        mock_service = AsyncMock()
        mock_service.execute = AsyncMock(side_effect=ValueError("Test error"))
        
        with patch('app.services.llm.service.ModelServiceFactory.create_service') as mock_factory:
            mock_factory.return_value = mock_service
            
            with pytest.raises(ValueError, match="Test error"):
                await service.execute(sample_request)


class TestGetModelService:
    """Tests for get_model_service function."""
    
    def test_returns_service_instance(self):
        """Should return ModelService instance."""
        service = get_model_service()
        assert isinstance(service, ModelService)
    
    def test_returns_singleton(self):
        """Should return same instance (singleton pattern)."""
        service1 = get_model_service()
        service2 = get_model_service()
        assert service1 is service2

