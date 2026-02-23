"""
Unit tests for API dependencies.
"""
import pytest
from app.api.dependencies import get_model_service
from app.services.llm.service import ModelService


class TestDependencies:
    """Tests for API dependencies."""
    
    def test_get_model_service_returns_service(self):
        """Should return ModelService instance."""
        service = get_model_service()
        assert isinstance(service, ModelService)
    
    def test_get_model_service_singleton(self):
        """Should return same instance (singleton)."""
        service1 = get_model_service()
        service2 = get_model_service()
        assert service1 is service2

