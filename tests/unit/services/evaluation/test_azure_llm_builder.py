"""
Unit tests for Azure LLM builder.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.evaluation.llm_builders.azure_llm_builder import AzureLLMBuilder


class TestAzureLLMBuilder:
    """Tests for AzureLLMBuilder."""
    
    def test_build_client(self):
        """Should build Azure OpenAI client."""
        builder = AzureLLMBuilder(
            {"modelName": "gpt-4", "apiKey": "test-key"},
            {"endpoint": "https://test.openai.azure.com", "apiVersion": "2024-02-15-preview"}
        )
        
        with patch('app.services.evaluation.llm_builders.azure_llm_builder.create_async_azure_openai_client') as mock_client:
            mock_client.return_value = Mock()
            
            client = builder.build_client()
            
            assert client is not None
            mock_client.assert_called_once()
    
    def test_get_provider(self):
        """Should return 'openai' for provider."""
        builder = AzureLLMBuilder(
            {"modelName": "gpt-4", "apiKey": "test-key"},
            {"endpoint": "https://test.openai.azure.com"}
        )
        
        assert builder.get_provider() == "openai"
    
    def test_build_with_deployment_name(self):
        """Should build with deployment name from config."""
        builder = AzureLLMBuilder(
            {"modelName": "gpt-4", "apiKey": "test-key"},
            {
                "endpoint": "https://test.openai.azure.com",
                "apiVersion": "2024-02-15-preview",
                "deploymentName": "gpt-4-deployment"
            }
        )
        
        with patch('app.services.evaluation.llm_builders.azure_llm_builder.create_async_azure_openai_client') as mock_client:
            with patch('ragas.llms.llm_factory') as mock_factory:
                mock_client.return_value = Mock()
                mock_factory.return_value = Mock()
                
                result = builder.build()
                
                assert result is not None
                mock_client.assert_called_once()

