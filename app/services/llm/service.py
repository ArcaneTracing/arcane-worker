"""
Main LLM service that routes requests to provider-specific services.

Uses the Factory pattern to create appropriate service instances
based on the adapter type (OpenAI, Anthropic, Azure, etc.).
Caches built LLM clients in-memory to avoid repeated instantiation
when processing many jobs with the same model configuration.
"""
from typing import Dict, Any, Optional
from app.config import settings
from app.core.cache import TTLCache
from app.models.schemas import LLMServiceRequestDto
from app.services.llm.base import BaseModelService
from app.services.llm.factory import ModelServiceFactory


class ModelService:
    """Service to route requests to appropriate model services using factory pattern."""

    def __init__(self) -> None:
        self._service_cache = TTLCache(
            ttl_seconds=settings.CACHE_MODEL_SERVICE_TTL,
            max_size=settings.CACHE_MODEL_SERVICE_MAX_SIZE,
        )

    async def execute(self, request: LLMServiceRequestDto) -> Dict[str, Any]:
        """
        Execute LLM request with full request DTO.

        Uses cached LLM client when available (keyed by model_configuration_id),
        otherwise creates via factory and caches for reuse.

        Returns a dictionary matching LLMServiceResponseDto structure.
        """
        config_id = request.model_configuration.id
        service: Optional[BaseModelService] = await self._service_cache.get(config_id)

        if service is None:
            api_key = request.model_configuration.configuration.api_key
            service = ModelServiceFactory.create_service(
                request.model_configuration.configuration,
                api_key,
            )
            await self._service_cache.set(config_id, service)

        return await service.execute(request)


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    Get or create the model service singleton.
    
    Returns:
        ModelService singleton instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

