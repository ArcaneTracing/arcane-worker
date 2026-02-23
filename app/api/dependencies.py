"""
FastAPI dependencies.
"""
from app.services.llm.service import get_model_service as _get_model_service


def get_model_service():
    """
    FastAPI dependency for getting the model service.
    
    This is a wrapper around the service's get_model_service function
    to maintain FastAPI dependency injection pattern.
    """
    return _get_model_service()

