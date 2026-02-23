"""
Unit tests for LLM client factories.
"""
import pytest
from anthropic import AsyncAnthropic
from app.services.llm.clients import (
    create_openai_client,
    create_azure_openai_client,
    create_anthropic_client,
    create_async_openai_client,
    create_async_azure_openai_client,
    create_async_anthropic_client,
)


class TestClientFactories:
    """Tests for client factory functions."""
    
    def test_create_openai_client(self):
        """Should create OpenAI client."""
        client = create_openai_client(api_key="test-key")
        
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_create_openai_client_with_base_url(self):
        """Should create OpenAI client with custom base URL."""
        client = create_openai_client(
            api_key="test-key",
            base_url="https://custom.openai.com"
        )
        
        assert client is not None
    
    def test_create_openai_client_with_organization(self):
        """Should create OpenAI client with organization."""
        client = create_openai_client(
            api_key="test-key",
            organization="org-123"
        )
        
        assert client is not None
    
    def test_create_azure_openai_client_with_azure_class(self):
        """Should create Azure OpenAI client using AzureOpenAI class."""
        client = create_azure_openai_client(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            use_azure_class=True
        )
        
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_create_azure_openai_client_with_openai_class(self):
        """Should create Azure OpenAI client using OpenAI class."""
        client = create_azure_openai_client(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            use_azure_class=False
        )
        
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_create_azure_openai_client_with_deployment(self):
        """Should create Azure OpenAI client with deployment name."""
        client = create_azure_openai_client(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview",
            deployment_name="gpt-4",
            use_azure_class=False
        )
        
        assert client is not None
    
    def test_create_anthropic_client(self):
        """Should create Anthropic client."""
        client = create_anthropic_client(api_key="test-key")
        
        assert client is not None
        assert hasattr(client, 'messages')
    
    def test_create_anthropic_client_with_base_url(self):
        """Should create Anthropic client with custom base URL."""
        client = create_anthropic_client(
            api_key="test-key",
            base_url="https://custom.anthropic.com"
        )
        
        assert client is not None
    
    def test_create_async_openai_client(self):
        """Should create async OpenAI client."""
        client = create_async_openai_client(api_key="test-key")
        
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_create_async_azure_openai_client(self):
        """Should create async Azure OpenAI client."""
        client = create_async_azure_openai_client(
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
            api_version="2024-02-15-preview"
        )
        
        assert client is not None
        assert hasattr(client, 'chat')
    
    def test_create_async_anthropic_client(self):
        """Should create async Anthropic client (patched to filter top_p for RAGAS compatibility)."""
        client = create_async_anthropic_client(api_key="test-key")
        
        assert client is not None
        assert hasattr(client, 'messages')
        assert isinstance(client, AsyncAnthropic)

