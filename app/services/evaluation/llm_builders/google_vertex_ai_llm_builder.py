"""
Builder for Google Vertex AI models.

Uses the new google-genai SDK (Option 1 from RAGAS Gemini integration guide).
"""
import json
import logging
from typing import TYPE_CHECKING, Optional
from google.oauth2 import service_account
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder

if TYPE_CHECKING:
    from ragas.llms import BaseRagasLLM

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _parse_credentials(credentials_json: str) -> dict:
    """Parse service account credentials from JSON string."""
    if not credentials_json or not credentials_json.strip():
        raise ValueError(
            "Vertex AI requires service account credentials JSON. "
            "Create one in GCP Console: IAM & Admin > Service Accounts > Keys."
        )
    raw = credentials_json.strip()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid credentials JSON: {e}. "
            "Paste the full service account JSON from GCP Console."
        ) from e
    if not isinstance(info, dict):
        raise ValueError("Credentials must be a JSON object.")
    return info


class GoogleVertexAILLMBuilder(BaseLLMBuilder):
    """Builder for Google Vertex AI models using google-genai SDK."""

    def build_client(self):
        """Build google-genai Client for Vertex AI for RAGAS."""
        from google import genai

        logger.debug(f"Building Google Vertex AI client for model: {self.model_name}")

        info = _parse_credentials(self.api_key)
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )

        project = self.config_dict.get("project") or info.get("project_id")
        if not project:
            raise ValueError(
                "Vertex AI requires project. Set it in the config or in the credentials JSON."
            )
        location = self.config_dict.get("location") or "us-central1"

        client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=location,
        )

        logger.debug("Successfully created Google Vertex AI client")
        return client

    def get_provider(self) -> Optional[str]:
        """Provider for RAGAS llm_factory (auto-detects instructor adapter)."""
        return "google"
