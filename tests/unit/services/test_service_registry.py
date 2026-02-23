"""
Unit tests for LLM service registry.
"""
import pytest
from app.services.llm.service_registry import (
    get_service_registry,
    register_all_services,
    _service_registry
)
from app.models.schemas import AdapterType, BaseModelConfiguration
from unittest.mock import Mock, MagicMock


class TestServiceRegistry:
    """Tests for service registry."""
    
    def test_registry_has_openai(self):
        """Should have OpenAI service registered."""
        registry = get_service_registry()
        assert registry.has(AdapterType.OPENAI.value)
    
    def test_registry_has_azure(self):
        """Should have Azure service registered."""
        registry = get_service_registry()
        assert registry.has(AdapterType.AZURE.value)
    
    def test_registry_has_anthropic(self):
        """Should have Anthropic service registered."""
        registry = get_service_registry()
        assert registry.has(AdapterType.ANTHROPIC.value)
    
    def test_create_openai_service(self):
        """Should create OpenAI service."""
        registry = get_service_registry()
        model_config = BaseModelConfiguration(
            adapter=AdapterType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        service = registry.create_or_raise(
            AdapterType.OPENAI.value,
            model_config=model_config,
            api_key="test-key"
        )
        
        assert service is not None
        assert service.get_adapter_type() == AdapterType.OPENAI.value
    
    def test_create_azure_service(self):
        """Should create Azure service."""
        registry = get_service_registry()
        model_config = BaseModelConfiguration(
            adapter=AdapterType.AZURE,
            model_name="gpt-4",
            api_key="test-key",
            config={"endpoint": "https://test.openai.azure.com"}
        )
        
        service = registry.create_or_raise(
            AdapterType.AZURE.value,
            model_config=model_config,
            api_key="test-key"
        )
        
        assert service is not None
        assert service.get_adapter_type() == AdapterType.AZURE.value
    
    def test_raises_for_missing_endpoint(self):
        """Should raise ValueError for Azure without endpoint."""
        registry = get_service_registry()
        model_config = BaseModelConfiguration(
            adapter=AdapterType.AZURE,
            model_name="gpt-4",
            api_key="test-key",
            config={}
        )
        
        with pytest.raises(ValueError, match="Azure endpoint is required"):
            registry.create_or_raise(
                AdapterType.AZURE.value,
                model_config=model_config,
                api_key="test-key"
            )

