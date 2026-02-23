"""
Unit tests for LLM builder factory.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.evaluation.llm_builders.llm_builder_factory import LLMBuilderFactory


class TestLLMBuilderFactory:
    """Tests for LLMBuilderFactory."""
    
    def test_create_builder_for_openai(self):
        """Should create OpenAI builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="openai",
            config={"modelName": "gpt-4", "apiKey": "test-key"},
            config_dict={}
        )
        
        assert builder is not None
        assert builder.model_name == "gpt-4"
    
    def test_create_builder_for_anthropic(self):
        """Should create Anthropic builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="anthropic",
            config={"modelName": "claude-3-opus-20240229", "apiKey": "test-key"},
            config_dict={}
        )
        
        assert builder is not None
        assert builder.model_name == "claude-3-opus-20240229"
    
    def test_create_builder_for_azure(self):
        """Should create Azure builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="azure",
            config={"modelName": "gpt-4", "apiKey": "test-key"},
            config_dict={"endpoint": "https://test.openai.azure.com"}
        )
        
        assert builder is not None
        assert builder.model_name == "gpt-4"
    
    def test_create_builder_raises_for_unsupported(self):
        """Should raise ValueError for unsupported adapter."""
        with pytest.raises(ValueError, match="Unknown key"):
            LLMBuilderFactory.create_builder(
                adapter="unsupported",
                config={"modelName": "test", "apiKey": "test-key"},
                config_dict={}
            )
    
    def test_create_builder_handles_errors(self):
        """Should log errors and re-raise."""
        with patch('app.services.evaluation.llm_builders.llm_builder_factory.get_builder_registry') as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.create_or_raise = Mock(side_effect=RuntimeError("Test error"))
            mock_registry.return_value = mock_registry_instance
            
            with pytest.raises(RuntimeError, match="Test error"):
                LLMBuilderFactory.create_builder(
                    adapter="openai",
                    config={"modelName": "gpt-4", "apiKey": "test-key"},
                    config_dict={}
                )
    
    def test_create_builder_for_bedrock(self):
        """Should create Bedrock builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="bedrock",
            config={"modelName": "anthropic.claude-3-opus-20240229-v1:0", "apiKey": "test-key"},
            config_dict={"region": "us-east-1"}
        )
        
        assert builder is not None
        assert builder.model_name == "anthropic.claude-3-opus-20240229-v1:0"
    
    def test_create_builder_for_google_vertex_ai(self):
        """Should create Google Vertex AI builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="google-vertex-ai",
            config={"modelName": "gemini-pro", "apiKey": "test-key"},
            config_dict={"project": "test-project", "location": "us-central1"}
        )
        
        assert builder is not None
        assert builder.model_name == "gemini-pro"
    
    def test_create_builder_for_google_ai_studio(self):
        """Should create Google AI Studio builder."""
        builder = LLMBuilderFactory.create_builder(
            adapter="google-ai-studio",
            config={"modelName": "gemini-pro", "apiKey": "test-key"},
            config_dict={}
        )
        
        assert builder is not None
        assert builder.model_name == "gemini-pro"

