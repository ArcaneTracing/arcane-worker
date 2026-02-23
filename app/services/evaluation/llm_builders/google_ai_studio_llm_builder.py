"""
Builder for Google AI Studio models.

Uses the new google-genai SDK (Option 1 from RAGAS Gemini integration guide).
"""
import logging
from typing import TYPE_CHECKING, Optional
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)


class GoogleAIStudioLLMBuilder(BaseLLMBuilder):
    """Builder for Google AI Studio models using google-genai SDK."""

    def build_client(self):
        """Build google-genai Client for RAGAS."""
        from google import genai

        logger.debug(f"Building Google AI Studio client for model: {self.model_name}")

        client = genai.Client(api_key=self.api_key)

        logger.debug("Successfully created Google AI Studio client")
        return client

    def get_provider(self) -> Optional[str]:
        """Provider for RAGAS llm_factory (auto-detects instructor adapter)."""
        return "google"
