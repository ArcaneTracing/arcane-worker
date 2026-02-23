"""
Unit tests for OpenAI response processor.
"""
import pytest
from app.services.llm.openai.response_processor import OpenAIResponseProcessor
from unittest.mock import Mock


class TestOpenAIResponseProcessor:
    """Tests for OpenAIResponseProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Response processor fixture."""
        return OpenAIResponseProcessor()
    
    @pytest.fixture
    def mock_response(self):
        """Mock OpenAI response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = "Test response"
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.usage = Mock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 20
        response.usage.total_tokens = 30
        return response
    
    def test_process_basic_response(self, processor, mock_response):
        """Should process basic response."""
        result = processor.process(mock_response, "gpt-4")
        
        assert result["output"] == "Test response"
        assert result["model"]["id"] == "gpt-4"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 30
    
    def test_process_with_tool_calls(self, processor, mock_response):
        """Should process response with tool calls."""
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function = Mock()
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"arg1": "value1"}'
        
        mock_response.choices[0].message.tool_calls = [tool_call]
        
        result = processor.process(mock_response, "gpt-4")
        
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "test_function"
        assert result["tool_calls"][0]["arguments"]["arg1"] == "value1"
    
    def test_process_with_string_arguments(self, processor, mock_response):
        """Should parse JSON string arguments."""
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function = Mock()
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"key": "value"}'
        
        mock_response.choices[0].message.tool_calls = [tool_call]
        
        result = processor.process(mock_response, "gpt-4")
        
        assert isinstance(result["tool_calls"][0]["arguments"], dict)
        assert result["tool_calls"][0]["arguments"]["key"] == "value"
    
    def test_process_without_usage(self, processor, mock_response):
        """Should handle response without usage."""
        del mock_response.usage
        
        result = processor.process(mock_response, "gpt-4")
        
        assert result["usage"] is None

