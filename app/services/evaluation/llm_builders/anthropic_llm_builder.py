"""
Builder for Anthropic models.

Note: Anthropic API does not allow both temperature and top_p for certain models.
We prefer temperature and exclude top_p when both are present.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional
from anthropic import AsyncAnthropic
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder
from app.services.llm.clients import create_async_anthropic_client

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


class AnthropicLLMBuilder(BaseLLMBuilder):
    """Builder for Anthropic models"""

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Build model kwargs for RAGAS. Anthropic does not allow both temperature and top_p.
        We exclude top_p when both are present. The create_async_anthropic_client returns
        a wrapped client that filters top_p from all API calls (RAGAS adds top_p=0.1 by default).
        """
        kwargs = super()._build_model_kwargs()
        if "temperature" in kwargs and "top_p" in kwargs:
            del kwargs["top_p"]
            logger.debug("Anthropic: excluded top_p (cannot use both temperature and top_p)")
        return kwargs

    def build_client(self) -> AsyncAnthropic:
        """Build async Anthropic client instance for RAGAS"""
        logger.debug(f"Building async Anthropic client for model: {self.model_name}")
        return create_async_anthropic_client(
            api_key=self.api_key,
            base_url=self.config_dict.get("baseUrl"),
            timeout=self.config_dict.get("timeout")
        )
    
    def get_provider(self) -> Optional[str]:
        """Get provider name for Anthropic"""
        return "anthropic"

