"""
Builder for OpenAI models.
"""
import logging
from typing import TYPE_CHECKING, Optional
from openai import AsyncOpenAI
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder
from app.services.llm.clients import create_async_openai_client

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


class OpenAILLMBuilder(BaseLLMBuilder):
    """Builder for OpenAI models"""
    
    def build_client(self) -> AsyncOpenAI:
        """Build async OpenAI client instance for RAGAS"""
        logger.debug(f"Building async OpenAI client for model: {self.model_name}")
        return create_async_openai_client(
            api_key=self.api_key,
            base_url=self.config_dict.get("baseUrl"),
            organization=self.config_dict.get("organization")
        )
    
    def get_provider(self) -> Optional[str]:
        """Get provider name for OpenAI"""
        return "openai"

