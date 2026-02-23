"""
Builder for creating embeddings instances from model configuration.
"""
import logging
from typing import Dict, Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding

from app.services.llm.clients import create_async_openai_client, create_async_azure_openai_client
from app.models.schemas import AdapterType

logger = logging.getLogger(__name__)


class EmbeddingsBuilder:
    """Builder for creating embeddings instances from model configuration."""
    
    def build_from_config(self, model_config: Dict[str, Any], embedding_model: Optional[str] = None) -> "BaseRagasEmbedding":
        """
        Build a ragas embeddings instance from model configuration.
        
        Args:
            model_config: Model configuration dictionary from API
            embedding_model: Optional embedding model name (defaults to provider-specific defaults)
            
        Returns:
            BaseRagasEmbedding instance for use with ragas
            
        Raises:
            ValueError: If adapter type is not supported or configuration is invalid
        """
        from ragas.embeddings.base import embedding_factory
        
        from app.core.validators import extract_config_section, validate_required_field
        
        logger.debug(f"build_from_config called. Model config top-level keys: {list(model_config.keys())}")
        
        # Extract and validate configuration section
        config = extract_config_section(model_config, "configuration", "Model configuration")
        
        # Extract and validate required fields
        adapter = config.get("adapter")
        validate_required_field(adapter, "Adapter type", "model configuration")
        
        api_key = config.get("apiKey")
        validate_required_field(api_key, "API key", "model configuration")
        
        config_dict = config.get("config", {})
        logger.info(f"Extracted config - Adapter: {adapter}, Has config dict: {bool(config_dict)}")
        
        try:
            # Build embeddings based on adapter type
            if adapter == AdapterType.OPENAI:
                return self._build_openai_embeddings(
                    api_key=api_key,
                    config_dict=config_dict,
                    embedding_model=embedding_model
                )
            elif adapter == AdapterType.AZURE:
                return self._build_azure_embeddings(
                    api_key=api_key,
                    config_dict=config_dict,
                    embedding_model=embedding_model
                )
            else:
                # For other adapters, try to use OpenAI embeddings as fallback
                # This is a common pattern since many providers support OpenAI-compatible embeddings
                logger.warning(f"Adapter {adapter} not explicitly supported for embeddings, using OpenAI embeddings")
                return self._build_openai_embeddings(
                    api_key=api_key,
                    config_dict=config_dict,
                    embedding_model=embedding_model
                )
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="EmbeddingsBuilder",
                context="build_from_config()",
                additional_info={"adapter": adapter}
            )
            raise
    
    def _build_openai_embeddings(
        self,
        api_key: str,
        config_dict: Dict[str, Any],
        embedding_model: Optional[str] = None
    ) -> "BaseRagasEmbedding":
        """Build OpenAI embeddings instance."""
        from ragas.embeddings.base import embedding_factory
        
        # Default embedding model for OpenAI
        model = embedding_model or config_dict.get("embeddingModel") or "text-embedding-3-small"
        
        logger.info(f"Building OpenAI embeddings - Model: {model}")
        
        client = create_async_openai_client(
            api_key=api_key,
            base_url=config_dict.get("baseUrl"),
            organization=config_dict.get("organization")
        )
        
        embeddings = embedding_factory("openai", model=model, client=client)
        
        logger.info(f"Successfully created OpenAI embeddings instance for model: {model}")
        return embeddings
    
    def _build_azure_embeddings(
        self,
        api_key: str,
        config_dict: Dict[str, Any],
        embedding_model: Optional[str] = None
    ) -> "BaseRagasEmbedding":
        """Build Azure OpenAI embeddings instance."""
        from ragas.embeddings.base import embedding_factory
        
        endpoint = config_dict.get("endpoint")
        if not endpoint:
            raise ValueError("Azure endpoint is required in config")
        
        api_version = config_dict.get("apiVersion", "2024-02-15-preview")
        # Use embedding_model if provided, otherwise try deploymentName from config, or default
        deployment_name = embedding_model or config_dict.get("deploymentName") or "text-embedding-3-small"
        
        logger.info(f"Building Azure OpenAI embeddings - Deployment: {deployment_name}")
        
        client = create_async_azure_openai_client(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name
        )
        
        # Azure uses OpenAI embeddings with custom client
        embeddings = embedding_factory("openai", model=deployment_name, client=client)
        
        logger.info(f"Successfully created Azure OpenAI embeddings instance for deployment: {deployment_name}")
        return embeddings

