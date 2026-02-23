"""
Unit tests for base LLM builder.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder


class TestBaseLLMBuilder:
    """Tests for BaseLLMBuilder abstract class."""
    
    def test_build_calls_build_client_and_llm_factory(self):
        """Should call build_client and llm_factory when building."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {"modelName": "claude-3-opus-20240229", "apiKey": "test-key"},
            {}
        )
        
        # Patch at the correct import location (inside the method)
        with patch('ragas.llms.llm_factory') as mock_factory:
            mock_llm = Mock()
            mock_factory.return_value = mock_llm
            
            with patch.object(builder, 'build_client') as mock_build_client:
                mock_client = Mock()
                mock_build_client.return_value = mock_client
                
                result = builder.build()
                
                assert result == mock_llm
                mock_build_client.assert_called_once()
                mock_factory.assert_called_once()
    
    def test_build_model_kwargs_includes_temperature(self):
        """Should include temperature in model kwargs if provided."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "temperature": 0.8
            },
            {}
        )
        
        kwargs = builder._build_model_kwargs()
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.8
    
    def test_build_model_kwargs_includes_max_tokens(self):
        """Should include max_tokens in model kwargs if provided."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "maxTokens": 2000
            },
            {}
        )
        
        kwargs = builder._build_model_kwargs()
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] == 2000
    
    def test_build_model_kwargs_excludes_top_p_for_anthropic(self):
        """Anthropic does not allow both temperature and top_p. We exclude top_p when both
        are present. The client wrapper also filters top_p from API calls (RAGAS default)."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "topP": 0.9,
                "temperature": 0.7,
            },
            {}
        )
        
        kwargs = builder._build_model_kwargs()
        assert kwargs["temperature"] == 0.7
        assert "top_p" not in kwargs  # Excluded when temperature is present
    
    def test_build_model_kwargs_includes_stop_sequences(self):
        """Should include stop sequences in model kwargs if provided."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "stopSequences": ["\n\n", "END"]
            },
            {}
        )
        
        kwargs = builder._build_model_kwargs()
        assert "stop" in kwargs
        assert kwargs["stop"] == ["\n\n", "END"]
    
    def test_build_handles_errors(self):
        """Should log errors and re-raise during build."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {"modelName": "claude-3-opus-20240229", "apiKey": "test-key"},
            {}
        )
        
        with patch.object(builder, 'build_client', side_effect=ValueError("Test error")):
            with pytest.raises(ValueError, match="Test error"):
                builder.build()
    
    def test_build_model_kwargs_validates_temperature_range(self):
        """Should only include temperature if within valid range (0-2)."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        # Test temperature too high
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "temperature": 3.0  # Invalid, too high
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "temperature" not in kwargs
        
        # Test temperature too low
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "temperature": -1.0  # Invalid, too low
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "temperature" not in kwargs
    
    def test_build_model_kwargs_validates_max_tokens_minimum(self):
        """Should only include max_tokens if >= 1."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "maxTokens": 0  # Invalid, too low
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "max_tokens" not in kwargs
    
    def test_build_model_kwargs_validates_top_p_range(self):
        """Invalid top_p from config is excluded. Client wrapper filters top_p from API calls."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        # Test top_p too high - excluded by base validation
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "topP": 1.5  # Invalid, too high
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "top_p" not in kwargs  # Invalid range excluded by base
    
    def test_build_model_kwargs_validates_penalty_ranges(self):
        """Should only include penalties if within valid range (-2 to 2)."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        # Test frequency penalty too high
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "frequencyPenalty": 3.0  # Invalid, too high
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "frequency_penalty" not in kwargs
        
        # Test presence penalty too low
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "presencePenalty": -3.0  # Invalid, too low
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "presence_penalty" not in kwargs
    
    def test_build_model_kwargs_validates_stop_sequences(self):
        """Should only include stop sequences if non-empty list."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        # Test empty list
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "stopSequences": []  # Invalid, empty
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "stop" not in kwargs
        
        # Test non-list
        builder = AnthropicLLMBuilder(
            {
                "modelName": "claude-3-opus-20240229",
                "apiKey": "test-key",
                "stopSequences": "not-a-list"  # Invalid, not a list
            },
            {}
        )
        kwargs = builder._build_model_kwargs()
        assert "stop" not in kwargs
    
    def test_init_validates_required_fields(self):
        """Should validate required fields during initialization."""
        from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
        
        # Missing modelName
        with pytest.raises(Exception):  # ValidationError
            AnthropicLLMBuilder(
                {"apiKey": "test-key"},
                {}
            )
        
        # Missing apiKey
        with pytest.raises(Exception):  # ValidationError
            AnthropicLLMBuilder(
                {"modelName": "claude-3-opus-20240229"},
                {}
            )

