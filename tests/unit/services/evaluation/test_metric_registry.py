"""
Unit tests for RAGAS metric registry.
"""
import pytest
from app.services.evaluation.ragas.metric_registry import (
    MetricRegistry,
    get_registry,
    register_metric
)
from app.services.evaluation.ragas.base_metric import BaseEvaluationMetric
from app.services.evaluation.ragas.metric_ids import METRIC_IDS
from unittest.mock import Mock, AsyncMock
from pydantic import BaseModel


class TestInput(BaseModel):
    """Test input model."""
    value: str


class TestMetric(BaseEvaluationMetric[TestInput]):
    """Test metric implementation."""
    
    async def evaluate(self, input_data, llm=None, embeddings=None):
        """Test evaluate method."""
        from app.services.evaluation.ragas.models.ragas_score import RagasScore
        return RagasScore(score=0.5, metric="test", id="test-id")


class TestMetricRegistry:
    """Tests for MetricRegistry."""
    
    def test_register_and_get_metric(self):
        """Should register and retrieve metric."""
        registry = MetricRegistry()
        metric = TestMetric(
            metric_id="test-id",
            metric_name="TestMetric",
            input_class=TestInput
        )
        
        registry.register(metric)
        retrieved = registry.get("test-id")
        
        assert retrieved is not None
        assert retrieved.metric_id == "test-id"
        assert retrieved.metric_name == "TestMetric"
    
    def test_has_checks_metric_existence(self):
        """Should check if metric exists."""
        registry = MetricRegistry()
        metric = TestMetric(
            metric_id="test-id",
            metric_name="TestMetric",
            input_class=TestInput
        )
        
        registry.register(metric)
        assert registry.has("test-id") is True
        assert registry.has("missing-id") is False
    
    def test_get_all_metric_ids(self):
        """Should return all registered metric IDs."""
        registry = MetricRegistry()
        metric1 = TestMetric(
            metric_id="test-id-1",
            metric_name="TestMetric1",
            input_class=TestInput
        )
        metric2 = TestMetric(
            metric_id="test-id-2",
            metric_name="TestMetric2",
            input_class=TestInput
        )
        
        registry.register(metric1)
        registry.register(metric2)
        
        ids = registry.get_all_metric_ids()
        assert "test-id-1" in ids
        assert "test-id-2" in ids
    
    def test_registry_has_context_precision(self):
        """Should have context_precision metric registered."""
        registry = get_registry()
        assert registry.has(METRIC_IDS["context_precision"])
    
    def test_registry_has_multiple_metrics(self):
        """Should have multiple metrics registered."""
        registry = get_registry()
        assert registry.has(METRIC_IDS["context_precision"])
        assert registry.has(METRIC_IDS["answer_relevancy"])
        assert registry.has(METRIC_IDS["faithfulness"])

