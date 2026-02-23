"""
Unit tests for RAGAS processor.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.evaluation.ragas.ragas_processor import RagasProcessor
from app.services.evaluation.ragas.metric_registry import get_registry
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class TestRagasProcessor:
    """Tests for RagasProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Processor fixture with mocked dependencies."""
        processor = RagasProcessor()
        processor.model_config_client = AsyncMock()
        processor.llm_builder = Mock()
        processor.embeddings_builder = Mock()
        processor.llm_builder.build_from_config = Mock(return_value=Mock())
        processor.embeddings_builder.build_from_config = Mock(return_value=Mock())
        return processor
    
    @pytest.mark.asyncio
    async def test_evaluate_with_registered_metric(self, processor):
        """Should evaluate using registered metric."""
        from app.services.evaluation.ragas.models.ragas_score import RagasScore
        from pydantic import BaseModel
        
        # Create a proper mock metric with validate_input method
        class TestInput(BaseModel):
            test: str
        
        mock_metric = Mock()
        mock_result = RagasScore(score=0.85, metric="test", id="test-id")
        mock_metric.evaluate = AsyncMock(return_value=mock_result)
        mock_metric.requires_llm = False
        mock_metric.requires_embeddings = False
        mock_metric.validate_input = Mock(return_value=TestInput(test="data"))
        
        # Mock registry to return our mock metric
        processor.registry = Mock()
        processor.registry.get = Mock(return_value=mock_metric)
        
        result = await processor.evaluate(
            ragas_score_key=METRIC_IDS["context_precision"],
            score_mapping={"test": "data"},
            ragas_model_config_id="model-1"
        )
        
        assert result.score == 0.85
        assert mock_metric.validate_input.called
    
    @pytest.mark.asyncio
    async def test_raises_for_unknown_metric(self, processor):
        """Should raise ValueError for unknown metric."""
        with patch('app.services.evaluation.ragas.ragas_processor.get_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.get = Mock(return_value=None)
            mock_get_registry.return_value = mock_registry
            
            with pytest.raises(ValueError, match="Unknown ragasScoreKey"):
                await processor.evaluate(
                    ragas_score_key="unknown-key",
                    score_mapping={},
                    ragas_model_config_id="model-1"
                )

