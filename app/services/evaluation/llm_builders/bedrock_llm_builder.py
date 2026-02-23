"""
Builder for AWS Bedrock models.
"""
import logging
from typing import TYPE_CHECKING, Optional
from openai import AsyncOpenAI
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


class BedrockLLMBuilder(BaseLLMBuilder):
    """Builder for AWS Bedrock models"""
    
    def build_client(self) -> AsyncOpenAI:
        """Build async AWS Bedrock client instance using LiteLLM proxy for RAGAS"""
        logger.debug(f"Building async AWS Bedrock client for model: {self.model_name}")
        
        # Bedrock typically uses LiteLLM proxy
        # Check if a proxy endpoint is provided, otherwise assume default LiteLLM proxy
        endpoint_url = self.config_dict.get("endpointUrl", "http://0.0.0.0:4000")
        
        logger.debug(f"Using Bedrock endpoint: {endpoint_url}")
        
        # Use async OpenAI client compatible with LiteLLM proxy
        client_kwargs = {
            "api_key": self.api_key or "bedrock",  # Bedrock uses IAM auth, but LiteLLM proxy may need a placeholder
            "base_url": endpoint_url,
        }
        
        client = AsyncOpenAI(**client_kwargs)
        logger.debug("Successfully created async AWS Bedrock client (via LiteLLM proxy)")
        return client
    
    def get_provider(self) -> Optional[str]:
        """Get provider name for AWS Bedrock"""
        # Bedrock uses LiteLLM adapter, which requires explicit adapter selection
        # We'll return None and let the factory handle adapter selection
        return None
    
    def build(self) -> "BaseRagasLLM":
        """Build and return the Ragas LLM instance using llm_factory with litellm adapter"""
        from ragas.llms import llm_factory
        
        logger.info(f"Building Ragas LLM for Bedrock - Model: {self.model_name}")
        try:
            client = self.build_client()
            model_kwargs = self._build_model_kwargs()
            
            logger.debug(f"Calling llm_factory with model={self.model_name}, adapter=litellm")
            # Bedrock requires explicit litellm adapter
            llm = llm_factory(
                self.model_name,
                client=client,
                adapter="litellm",
                **model_kwargs
            )
            
            logger.info(f"Successfully created Ragas LLM instance for Bedrock model: {self.model_name}")
            return llm
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="BedrockLLMBuilder",
                context="build()",
                additional_info={"model": self.model_name}
            )
            raise

