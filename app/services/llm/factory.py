"""
Factory for creating model services based on adapter type.
Uses registry pattern for dynamic service creation.
"""
from app.models.schemas import BaseModelConfiguration
from app.services.llm.base import BaseModelService
from app.services.llm.service_registry import get_service_registry

# Import to ensure services are registered
from app.services.llm import service_registry  # noqa: F401


class ModelServiceFactory:
    """
    Factory for creating model services based on adapter type.
    
    Uses registry pattern for dynamic service creation.
    """
    
    @staticmethod
    def create_service(
        model_config: BaseModelConfiguration,
        api_key: str
    ) -> BaseModelService:
        """
        Create the appropriate model service based on adapter type.
        
        Args:
            model_config: Model configuration containing adapter type and settings
            api_key: API key for authentication
            
        Returns:
            Appropriate model service instance
            
        Raises:
            ValueError: If adapter type is not supported
        """
        adapter = model_config.adapter.value if hasattr(model_config.adapter, 'value') else str(model_config.adapter)
        registry = get_service_registry()
        
        return registry.create_or_raise(
            adapter,
            model_config=model_config,
            api_key=api_key
        )

