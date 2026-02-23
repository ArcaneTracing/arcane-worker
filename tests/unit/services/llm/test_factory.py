"""
Unit tests for LLM service factory.
"""
import pytest
from app.services.llm.factory import ModelServiceFactory
from app.models.schemas import AdapterType, BaseModelConfiguration


class TestModelServiceFactory:
    """Tests for ModelServiceFactory."""
    
    def test_create_openai_service(self):
        """Should create OpenAI service."""
        model_config = BaseModelConfiguration(
            adapter=AdapterType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        service = ModelServiceFactory.create_service(
            model_config=model_config,
            api_key="test-key"
        )
        
        assert service is not None
        assert service.get_adapter_type() == AdapterType.OPENAI.value
    
    def test_create_azure_service(self):
        """Should create Azure service."""
        model_config = BaseModelConfiguration(
            adapter=AdapterType.AZURE,
            model_name="gpt-4",
            api_key="test-key",
            config={"endpoint": "https://test.openai.azure.com"}
        )
        
        service = ModelServiceFactory.create_service(
            model_config=model_config,
            api_key="test-key"
        )
        
        assert service is not None
        assert service.get_adapter_type() == AdapterType.AZURE.value
    
    def test_create_anthropic_service(self):
        """Should create Anthropic service."""
        model_config = BaseModelConfiguration(
            adapter=AdapterType.ANTHROPIC,
            model_name="claude-3",
            api_key="test-key"
        )
        
        service = ModelServiceFactory.create_service(
            model_config=model_config,
            api_key="test-key"
        )
        
        assert service is not None
        assert service.get_adapter_type() == AdapterType.ANTHROPIC.value
    
    def test_raises_for_unsupported_adapter(self):
        """Should raise ValueError for unsupported adapter (test via registry)."""
        from app.services.llm.service_registry import get_service_registry
        
        # Test that registry raises for unsupported adapter
        registry = get_service_registry()
        with pytest.raises(ValueError, match="Unknown key"):
            # Directly test registry with unsupported key
            registry.create_or_raise(
                "unsupported-adapter",
                model_config=BaseModelConfiguration(
                    adapter=AdapterType.OPENAI,
                    model_name="test",
                    api_key="test-key"
                ),
                api_key="test-key"
            )

