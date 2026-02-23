"""
Base class for LLM builders.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


class BaseLLMBuilder(ABC):
    """Base class for LLM builders"""
    
    def __init__(self, config: Dict[str, Any], config_dict: Dict[str, Any]):
        """
        Initialize the builder with configuration.
        
        Args:
            config: Main configuration dictionary
            config_dict: Provider-specific config dictionary
        """
        logger.debug(f"Initializing {self.__class__.__name__} with config keys: {list(config.keys())}")
        self.config = config
        self.config_dict = config_dict
        self.model_name = config.get("modelName")
        self.api_key = config.get("apiKey")
        
        from app.core.validators import validate_required_field
        
        logger.debug(f"Model name: {self.model_name}, API key present: {bool(self.api_key)}")
        
        validate_required_field(self.model_name, "Model name", "model configuration")
        validate_required_field(self.api_key, "API key", "model configuration")
        
        logger.debug(f"Successfully initialized {self.__class__.__name__}")
    
    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Build model kwargs for ragas llm_factory"""
        logger.debug(f"Building model kwargs for {self.__class__.__name__}")
        kwargs = {}
        
        # Add optional parameters if present and valid
        # Temperature: 0-2 (OpenAI accepts 0-2)
        temperature = self.config.get("temperature")
        if temperature is not None and 0 <= temperature <= 2:
            kwargs["temperature"] = temperature
        
        # Max tokens: must be >= 1 (OpenAI requirement)
        max_tokens = self.config.get("maxTokens")
        if max_tokens is not None and max_tokens >= 1:
            kwargs["max_tokens"] = max_tokens
        
        # Top P: 0-1 (OpenAI accepts 0-1)
        top_p = self.config.get("topP")
        if top_p is not None and 0 <= top_p <= 1:
            kwargs["top_p"] = top_p
        
        # Frequency penalty: -2 to 2 (OpenAI accepts -2 to 2)
        frequency_penalty = self.config.get("frequencyPenalty")
        if frequency_penalty is not None and -2 <= frequency_penalty <= 2:
            kwargs["frequency_penalty"] = frequency_penalty
        
        # Presence penalty: -2 to 2 (OpenAI accepts -2 to 2)
        presence_penalty = self.config.get("presencePenalty")
        if presence_penalty is not None and -2 <= presence_penalty <= 2:
            kwargs["presence_penalty"] = presence_penalty
        
        # Stop sequences: must be a non-empty list
        stop_sequences = self.config.get("stopSequences")
        if stop_sequences and isinstance(stop_sequences, list) and len(stop_sequences) > 0:
            kwargs["stop"] = stop_sequences
        
        logger.debug(f"Model kwargs built: {list(kwargs.keys())}")
        return kwargs
    
    @abstractmethod
    def build_client(self) -> Any:
        """Build and return the provider client instance (e.g., OpenAI, Anthropic)"""
        pass
    
    @abstractmethod
    def get_provider(self) -> Optional[str]:
        """Get the provider name for ragas llm_factory (e.g., 'openai', 'anthropic', 'google')"""
        pass
    
    def build(self) -> "BaseRagasLLM":
        """Build and return the Ragas LLM instance using llm_factory"""
        from ragas.llms import llm_factory
        
        logger.info(f"Building Ragas LLM - Model: {self.model_name}")
        try:
            client = self.build_client()
            provider = self.get_provider()
            model_kwargs = self._build_model_kwargs()
            
            # Build llm_factory kwargs
            factory_kwargs = {
                "client": client,
                **model_kwargs
            }
            
            # Only add provider if it's not None
            if provider is not None:
                factory_kwargs["provider"] = provider
            
            logger.debug(f"Calling llm_factory with model={self.model_name}, provider={provider}")
            llm = llm_factory(
                self.model_name,
                **factory_kwargs
            )
            
            logger.info(f"Successfully created Ragas LLM instance for model: {self.model_name}")
            return llm
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name=self.__class__.__name__,
                context="build()",
                additional_info={"model": self.model_name}
            )
            raise

