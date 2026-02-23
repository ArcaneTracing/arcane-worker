"""
Unit tests for experiment processor.
"""
import pytest
from unittest.mock import AsyncMock
from app.domain.experiment.processor import ExperimentJobProcessor
from app.models.schemas import (
    AdapterType,
    ExperimentJobDto,
    TemplateFormat,
    TemplateType,
)


class TestExperimentJobProcessor:
    """Tests for ExperimentJobProcessor."""

    @pytest.fixture
    def processor(self):
        """Processor fixture with mocked dependencies."""
        processor = ExperimentJobProcessor()
        processor.model_service = AsyncMock()
        processor.model_service.execute = AsyncMock(
            return_value={"output": "test response", "usage": {"total_tokens": 100}}
        )
        processor.model_config_client = AsyncMock()
        processor.model_config_client.fetch_model_config = AsyncMock(
            return_value={
                "id": "config-1",
                "name": "Test Config",
                "configuration": {
                    "adapter": AdapterType.OPENAI,
                    "modelName": "gpt-4",
                    "apiKey": "test-key",
                },
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        )
        processor.prompt_version_client = AsyncMock()
        processor.prompt_version_client.fetch_latest_version = AsyncMock(
            return_value={
                "id": "pv-1",
                "promptId": "prompt-1",
                "modelConfigurationId": "config-1",
                "template": {"type": "string", "template": "Evaluate: {input}"},
                "templateType": TemplateType.STR,
                "templateFormat": TemplateFormat.F_STRING,
                "invocationParameters": {"type": "openai", "openai": {}},
            }
        )
        return processor

    @pytest.fixture
    def sample_job_data(self):
        """Sample experiment job DTO with promptId."""
        return ExperimentJobDto(
            experiment_id="exp-1",
            dataset_row_id="row-1",
            prompt_id="prompt-1",
            inputs={"question": "What is the capital of France?"},
        )
    
    @pytest.mark.asyncio
    async def test_process_experiment_job(self, processor, sample_job_data):
        """Should process experiment job."""
        result = await processor.process(sample_job_data)
        
        assert result is not None
        assert result.experiment_id == "exp-1"
        assert result.dataset_row_id == "row-1"
        assert processor.model_service.execute.called
    
    @pytest.mark.asyncio
    async def test_process_handles_errors(self, processor, sample_job_data):
        """Should handle errors during processing."""
        processor.model_service.execute = AsyncMock(side_effect=ValueError("Test error"))
        
        result = await processor.process(sample_job_data)
        
        # Should return error result
        assert result is not None
        assert result.error is not None
        assert result.experiment_id == "exp-1"
    
    @pytest.mark.asyncio
    async def test_process_includes_metadata(self, processor, sample_job_data):
        """Should include metadata in result."""
        result = await processor.process(sample_job_data)
        
        assert result.metadata is not None
        assert "execution_time_ms" in result.metadata
        assert result.metadata.get("tokens_used") == 100

    @pytest.mark.asyncio
    async def test_process_passes_through_message_id(self, processor, sample_job_data):
        """Should pass message_id from job to result."""
        sample_job_data.message_id = "exp-1-row-1-1700000000000"
        result = await processor.process(sample_job_data)
        assert result.message_id == "exp-1-row-1-1700000000000"

    @pytest.mark.asyncio
    async def test_process_handles_errors_passes_through_message_id(self, processor, sample_job_data):
        """Should pass message_id through on error result."""
        sample_job_data.message_id = "exp-1-row-1-1700000000000"
        processor.model_service.execute = AsyncMock(side_effect=ValueError("Test error"))
        result = await processor.process(sample_job_data)
        assert result.message_id == "exp-1-row-1-1700000000000"
