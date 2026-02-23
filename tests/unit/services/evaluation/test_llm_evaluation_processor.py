"""
Unit tests for LLM evaluation processor.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.evaluation.llm_evaluation_processor import LLMEvaluationProcessor


class TestLLMEvaluationProcessor:
    """Tests for LLMEvaluationProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture with mocked dependencies."""
        processor = LLMEvaluationProcessor()
        processor.model_service = AsyncMock()
        processor.model_service.execute = AsyncMock(return_value={"output": "0.85"})
        return processor
    
    @pytest.fixture
    def sample_model_config(self, sample_model_configuration_with_key):
        """Sample model configuration."""
        return sample_model_configuration_with_key
    
    @pytest.fixture
    def sample_prompt_version(self, sample_prompt_version):
        """Sample prompt version."""
        return sample_prompt_version
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, processor, sample_model_config, sample_prompt_version):
        """Should evaluate successfully."""
        inputs = {"input": "test input"}
        
        result = await processor.evaluate(
            model_configuration=sample_model_config,
            prompt_version=sample_prompt_version,
            inputs=inputs
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert processor.model_service.execute.called
    
    @pytest.mark.asyncio
    async def test_evaluate_handles_errors(self, processor, sample_model_config, sample_prompt_version):
        """Should handle errors during evaluation."""
        processor.model_service.execute = AsyncMock(side_effect=ValueError("Test error"))
        
        inputs = {"input": "test input"}
        
        # Should re-raise error (not caught internally)
        with pytest.raises(ValueError, match="Test error"):
            await processor.evaluate(
                model_configuration=sample_model_config,
                prompt_version=sample_prompt_version,
                inputs=inputs
            )
    
    @pytest.mark.asyncio
    async def test_evaluate_returns_output_string(self, processor, sample_model_config, sample_prompt_version):
        """Should return output as string."""
        processor.model_service.execute = AsyncMock(return_value={"output": "0.85"})
        
        inputs = {"input": "test input"}
        
        result = await processor.evaluate(
            model_configuration=sample_model_config,
            prompt_version=sample_prompt_version,
            inputs=inputs
        )
        
        # Should return output string
        assert result is not None
        assert isinstance(result, str)
        assert result == "0.85"

