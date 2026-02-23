"""
Unit tests for LLM builder.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.evaluation.llm_builders.llm_builder import LLMBuilder
from app.services.evaluation.llm_builders.llm_builder_factory import LLMBuilderFactory


class TestLLMBuilder:
    """Tests for LLMBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Builder fixture."""
        return LLMBuilder()
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration."""
        return {
            "configuration": {
                "adapter": "openai",
                "modelName": "gpt-4",
                "apiKey": "test-key",
                "config": {}
            }
        }
    
    def test_build_from_config_success(self, builder, sample_model_config):
        """Should build LLM from configuration."""
        with patch.object(LLMBuilderFactory, 'create_builder') as mock_create:
            mock_builder_instance = Mock()
            mock_llm = Mock()
            mock_builder_instance.build = Mock(return_value=mock_llm)
            mock_create.return_value = mock_builder_instance
            
            result = builder.build_from_config(sample_model_config)
            
            assert result == mock_llm
            mock_create.assert_called_once()
            mock_builder_instance.build.assert_called_once()
    
    def test_build_from_config_missing_configuration(self, builder):
        """Should raise ValueError for missing configuration."""
        model_config = {}
        
        with pytest.raises(Exception):  # Can be ValidationError or ValueError
            builder.build_from_config(model_config)
    
    def test_build_from_config_missing_adapter(self, builder):
        """Should raise ValueError for missing adapter."""
        model_config = {
            "configuration": {
                "modelName": "gpt-4",
                "apiKey": "test-key"
            }
        }
        
        with pytest.raises(Exception):  # Can be ValidationError or ValueError
            builder.build_from_config(model_config)
    
    def test_build_from_config_calls_factory(self, builder, sample_model_config):
        """Should call factory to create builder."""
        with patch.object(LLMBuilderFactory, 'create_builder') as mock_create:
            mock_builder_instance = Mock()
            mock_builder_instance.build = Mock(return_value=Mock())
            mock_create.return_value = mock_builder_instance
            
            builder.build_from_config(sample_model_config)
            
            mock_create.assert_called_once()
            args = mock_create.call_args[0]
            assert args[0] == "openai"  # adapter
            assert "modelName" in args[1]  # config

