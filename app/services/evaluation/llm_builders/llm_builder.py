"""
Builder for creating LLM instances from model configuration.
"""
import json
import logging
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

from app.services.evaluation.llm_builders.llm_builder_factory import LLMBuilderFactory

logger = logging.getLogger(__name__)


class LLMBuilder:
    """Builder for creating LLM instances from model configuration."""
    
    def __init__(self, factory: type[LLMBuilderFactory] = None):
        """
        Initialize the builder with an optional factory class.
        
        Args:
            factory: Optional factory class (uses LLMBuilderFactory if not provided)
        """
        self.factory = factory or LLMBuilderFactory
    
    def build_from_config(self, model_config: Dict[str, Any]) -> "BaseRagasLLM":
        """
        Build a ragas LLM instance from model configuration.
        
        Args:
            model_config: Model configuration dictionary from API
            
        Returns:
            BaseRagasLLM instance for use with ragas
            
        Raises:
            ValueError: If adapter type is not supported or configuration is invalid
        """
        from app.core.validators import extract_config_section, validate_required_field
        
        logger.debug(f"build_from_config called. Model config top-level keys: {list(model_config.keys())}")
        
        # Extract and validate configuration section
        config = extract_config_section(model_config, "configuration", "Model configuration")
        
        # Extract adapter and validate
        adapter = config.get("adapter")
        validate_required_field(adapter, "Adapter type", "model configuration")
        
        # Extract other config values
        config_dict = config.get("config", {})
        # log config_dict as json
        logger.info(f"Config dict: {json.dumps(config_dict, indent=4)}")
        model_name = config.get("modelName")
        
        logger.info(f"Extracted config - Adapter: {adapter}, Model: {model_name}, Has config dict: {bool(config_dict)}")
        
        try:
            builder = self.factory.create_builder(adapter, config, config_dict)
            logger.info(f"Builder created, now building LLM instance for adapter: {adapter}")
            llm_instance = builder.build()
            logger.info(f"Successfully built LLM instance for adapter: {adapter}, model: {model_name}")
            return llm_instance
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="LLMBuilder",
                context="build_from_config()",
                additional_info={"adapter": adapter, "model": model_name}
            )
            raise

