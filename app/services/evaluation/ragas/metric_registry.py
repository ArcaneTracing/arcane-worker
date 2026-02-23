"""
Registry for RAGAS evaluation metrics.
Maps metric IDs to metric implementations.
"""
from typing import Dict, Optional
from app.services.evaluation.ragas.base_metric import BaseEvaluationMetric
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class MetricRegistry:
    """
    Registry for evaluation metrics.
    
    Provides a centralized way to register and retrieve metrics by their IDs.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._metrics: Dict[str, BaseEvaluationMetric] = {}
    
    def register(self, metric: BaseEvaluationMetric) -> None:
        """
        Register a metric in the registry.
        
        Args:
            metric: The metric instance to register
        """
        self._metrics[metric.metric_id] = metric
    
    def get(self, metric_id: str) -> Optional[BaseEvaluationMetric]:
        """
        Get a metric by its ID.
        
        Args:
            metric_id: The metric ID (UUID)
            
        Returns:
            The metric instance, or None if not found
        """
        return self._metrics.get(metric_id)
    
    def has(self, metric_id: str) -> bool:
        """
        Check if a metric is registered.
        
        Args:
            metric_id: The metric ID (UUID)
            
        Returns:
            True if the metric is registered, False otherwise
        """
        return metric_id in self._metrics
    
    def get_all_metric_ids(self) -> list[str]:
        """
        Get all registered metric IDs.
        
        Returns:
            List of all registered metric IDs
        """
        return list(self._metrics.keys())


# Global registry instance
_registry = MetricRegistry()


def get_registry() -> MetricRegistry:
    """
    Get the global metric registry.
    
    Returns:
        The global MetricRegistry instance
    """
    return _registry


def register_metric(metric: BaseEvaluationMetric) -> None:
    """
    Register a metric in the global registry.
    
    Args:
        metric: The metric instance to register
    """
    _registry.register(metric)

