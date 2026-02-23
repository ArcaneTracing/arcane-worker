"""
Factory for creating appropriate LLM builder instances.
Uses registry pattern for dynamic builder creation.
"""
import logging
from typing import Dict, Any
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder
from app.services.evaluation.llm_builders.builder_registry import get_builder_registry

# Import to ensure builders are registered
from app.services.evaluation.llm_builders import builder_registry  # noqa: F401

logger = logging.getLogger(__name__)


class LLMBuilderFactory:
    """
    Factory for creating appropriate LLM builder instances.
    
    Uses registry pattern for dynamic builder creation.
    """
    
    @classmethod
    def create_builder(
        cls,
        adapter: str,
        config: Dict[str, Any],
        config_dict: Dict[str, Any]
    ) -> BaseLLMBuilder:
        """
        Create the appropriate builder for the given adapter type.
        
        Args:
            adapter: Adapter type (e.g., "openai", "azure", etc.)
            config: Main configuration dictionary
            config_dict: Provider-specific config dictionary
            
        Returns:
            Appropriate LLM builder instance
            
        Raises:
            ValueError: If adapter type is not supported
        """
        logger.debug(f"Creating builder for adapter: {adapter}")
        logger.debug(f"Config keys: {list(config.keys())}, Config dict keys: {list(config_dict.keys())}")
        
        registry = get_builder_registry()
        
        try:
            builder = registry.create_or_raise(adapter, config, config_dict)
            logger.info(f"Successfully created builder for adapter: {adapter}")
            return builder
        except ValueError:
            # Re-raise ValueError for unsupported adapter types
            raise
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="LLMBuilderFactory",
                context="create_builder()",
                additional_info={"adapter": adapter}
            )
            raise

