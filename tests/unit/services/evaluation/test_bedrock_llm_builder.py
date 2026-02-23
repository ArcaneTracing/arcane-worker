"""
Unit tests for Bedrock LLM builder.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.evaluation.llm_builders.bedrock_llm_builder import BedrockLLMBuilder


class TestBedrockLLMBuilder:
    """Tests for BedrockLLMBuilder."""
    
    def test_build_client(self):
        """Should build Bedrock client using AsyncOpenAI."""
        from openai import AsyncOpenAI
        
        builder = BedrockLLMBuilder(
            {"modelName": "anthropic.claude-3-opus-20240229-v1:0", "apiKey": "test-key"},
            {"region": "us-east-1", "endpointUrl": "http://0.0.0.0:4000"}
        )
        
        with patch('app.services.evaluation.llm_builders.bedrock_llm_builder.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            client = builder.build_client()
            
            assert client is not None
            assert client == mock_client
            mock_openai.assert_called_once()
    
    def test_get_provider(self):
        """Should return None for provider."""
        builder = BedrockLLMBuilder(
            {"modelName": "anthropic.claude-3-opus-20240229-v1:0", "apiKey": "test-key"},
            {"region": "us-east-1"}
        )
        
        assert builder.get_provider() is None

