"""
Unit tests for evaluation job processor.
"""
import pytest
from unittest.mock import Mock, AsyncMock

from app.domain.evaluation.processor import EvaluationJobProcessor
from app.models.schemas import EvaluationJobDto


class TestEvaluationJobProcessor:
    """Tests for EvaluationJobProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture."""
        processor = EvaluationJobProcessor()
        processor.ragas_processor = AsyncMock()
        processor.llm_evaluation_processor = AsyncMock()
        return processor
    
    def _ragas_job(self, **overrides) -> EvaluationJobDto:
        """Create RAGAS evaluation job DTO."""
        base = {
            "evaluation_id": "eval-1",
            "score_id": "score-1",
            "scoring_type": "RAGAS",
            "dataset_row_id": "row-1",
            "experiment_result_id": "result-1",
            "ragas_score_key": "test-key",
            "ragas_model_configuration_id": "model-1",
            "score_mapping": {"test": "data"},
        }
        base.update(overrides)
        return EvaluationJobDto(**base)
    
    def _llm_job(self) -> EvaluationJobDto:
        """Create LLM evaluation job DTO."""
        return EvaluationJobDto(
            evaluation_id="eval-1",
            score_id="score-1",
            scoring_type="LLM",
            dataset_row_id="row-1",
            experiment_result_id="result-1",
            prompt_id="prompt-1",
            score_mapping={"test": "data"},
        )
    
    @pytest.mark.asyncio
    async def test_process_ragas_evaluation(self, processor):
        """Should process RAGAS evaluation."""
        mock_result = Mock()
        mock_result.score = 0.85
        mock_result.metric = "context_precision"
        mock_result.id = "test-id"
        
        processor.ragas_processor.evaluate = AsyncMock(return_value=mock_result)
        
        job_data = self._ragas_job()
        result = await processor.process(job_data)
        
        assert result.evaluation_id == "eval-1"
        assert result.score == "0.85"
        assert result.metric == "context_precision"
        assert processor.ragas_processor.evaluate.called
    
    @pytest.mark.asyncio
    async def test_process_llm_evaluation(self, processor):
        """Should process LLM-based evaluation (fetches prompt version and model config via API)."""
        processor.llm_evaluation_processor.evaluate = AsyncMock(return_value="0.75")
        processor.model_config_client = AsyncMock()
        processor.model_config_client.fetch_model_config = AsyncMock(
            return_value={
                "id": "config-1",
                "name": "Test Config",
                "configuration": {"adapter": "openai", "modelName": "gpt-4", "apiKey": "key"},
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
                "template": {"type": "string", "template": "test"},
                "templateType": "STR",
                "templateFormat": "F_STRING",
                "invocationParameters": {"type": "openai", "openai": {}},
            }
        )

        job_data = self._llm_job()
        result = await processor.process(job_data)

        assert result.evaluation_id == "eval-1"
        assert result.score == "0.75"
        assert processor.llm_evaluation_processor.evaluate.called
    
    @pytest.mark.asyncio
    async def test_handles_errors_gracefully(self, processor):
        """Should handle errors and return error result."""
        processor.ragas_processor.evaluate = AsyncMock(side_effect=ValueError("Test error"))
        
        job_data = self._ragas_job(score_mapping={})
        result = await processor.process(job_data)
        
        assert result.evaluation_id == "eval-1"
        assert result.error is not None or result.score is None
        assert result.score_id == "score-1"
    
    @pytest.mark.asyncio
    async def test_requires_ragas_score_key(self, processor):
        """Should raise error for RAGAS without score key."""
        job_data = self._ragas_job(ragas_score_key="", score_mapping={})
        
        result = await processor.process(job_data)
        assert result.error is not None or result.score is None

    @pytest.mark.asyncio
    async def test_passes_through_message_id(self, processor):
        """Should pass message_id from job to result."""
        job_data = self._ragas_job(message_id="eval-1-score-1-1700000000000")
        result = await processor.process(job_data)
        assert result.message_id == "eval-1-score-1-1700000000000"

