"""
Unit tests for embeddings builder.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.evaluation.embeddings_builder import EmbeddingsBuilder


class TestEmbeddingsBuilder:
    """Tests for EmbeddingsBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Builder fixture."""
        return EmbeddingsBuilder()
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration."""
        return {
            "configuration": {
                "adapter": "openai",
                "modelName": "text-embedding-ada-002",
                "apiKey": "test-key",
                "config": {}
            }
        }
    
    def test_build_from_config_success(self, builder, sample_model_config):
        """Should build embeddings from configuration."""
        with patch('app.services.evaluation.embeddings_builder.create_async_openai_client') as mock_create_client:
            with patch('ragas.embeddings.base.embedding_factory') as mock_embedding_factory:
                # Mock the embedding factory to return a mock embeddings instance
                mock_embeddings_instance = Mock()
                mock_embedding_factory.return_value = mock_embeddings_instance
                
                # Mock the client (not used directly but may be called)
                mock_client_instance = Mock()
                mock_create_client.return_value = mock_client_instance
                
                result = builder.build_from_config(sample_model_config)
                
                assert result == mock_embeddings_instance
                mock_embedding_factory.assert_called_once()
    
    def test_build_from_config_missing_configuration(self, builder):
        """Should raise ValidationError for missing configuration."""
        model_config = {}
        
        with pytest.raises(Exception):  # ValidationError
            builder.build_from_config(model_config)
    
    def test_build_from_config_missing_adapter(self, builder):
        """Should raise ValidationError for missing adapter."""
        model_config = {
            "configuration": {
                "modelName": "text-embedding-ada-002",
                "apiKey": "test-key"
            }
        }
        
        with pytest.raises(Exception):  # ValidationError
            builder.build_from_config(model_config)
    
    def test_build_from_config_missing_api_key(self, builder):
        """Should raise ValidationError for missing API key."""
        model_config = {
            "configuration": {
                "adapter": "openai",
                "modelName": "text-embedding-ada-002"
            }
        }
        
        with pytest.raises(Exception):  # ValidationError
            builder.build_from_config(model_config)

