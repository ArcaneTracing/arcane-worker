"""
Extended unit tests for embeddings builder (coverage improvement).
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.evaluation.embeddings_builder import EmbeddingsBuilder
from app.models.schemas import AdapterType


class TestEmbeddingsBuilderExtended:
    """Extended tests for EmbeddingsBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Builder fixture."""
        return EmbeddingsBuilder()
    
    def test_build_azure_embeddings(self, builder):
        """Should build Azure OpenAI embeddings."""
        sample_config = {
            "configuration": {
                "adapter": "azure",
                "modelName": "text-embedding-ada-002",
                "apiKey": "test-key",
                "config": {
                    "endpoint": "https://test.openai.azure.com",
                    "apiVersion": "2024-02-15-preview"
                }
            }
        }
        
        with patch('ragas.embeddings.base.embedding_factory') as mock_factory:
            with patch('app.services.evaluation.embeddings_builder.create_async_azure_openai_client') as mock_client:
                mock_embeddings = Mock()
                mock_factory.return_value = mock_embeddings
                mock_client.return_value = Mock()
                
                result = builder.build_from_config(sample_config)
                
                assert result == mock_embeddings
                mock_factory.assert_called_once()
    
    def test_build_fallback_to_openai_for_other_adapters(self, builder):
        """Should fall back to OpenAI embeddings for unsupported adapters."""
        sample_config = {
            "configuration": {
                "adapter": "anthropic",  # Not explicitly supported for embeddings
                "modelName": "claude-3",
                "apiKey": "test-key",
                "config": {}
            }
        }
        
        with patch('ragas.embeddings.base.embedding_factory') as mock_factory:
            with patch('app.services.evaluation.embeddings_builder.create_async_openai_client') as mock_client:
                mock_embeddings = Mock()
                mock_factory.return_value = mock_embeddings
                mock_client.return_value = Mock()
                
                result = builder.build_from_config(sample_config)
                
                assert result == mock_embeddings
                # Should use OpenAI as fallback
                mock_factory.assert_called_once()
    
    def test_build_handles_errors(self, builder):
        """Should log errors and re-raise."""
        sample_config = {
            "configuration": {
                "adapter": "openai",
                "modelName": "text-embedding-ada-002",
                "apiKey": "test-key",
                "config": {}
            }
        }
        
        with patch('ragas.embeddings.base.embedding_factory', side_effect=ValueError("Test error")):
            with pytest.raises(ValueError, match="Test error"):
                builder.build_from_config(sample_config)

