"""
Client for fetching model configurations from the API.

Uses in-memory TTL cache to avoid redundant API calls when processing
many jobs with the same model configuration.
"""
import logging
from typing import Dict, Any
import httpx
from app.config import settings
from app.core.cache import TTLCache

logger = logging.getLogger(__name__)


class ModelConfigClient:
    """Client for interacting with model configuration API."""

    def __init__(self) -> None:
        self._cache = TTLCache(ttl_seconds=settings.CACHE_MODEL_CONFIG_TTL)

    async def fetch_model_config(self, config_id: str) -> Dict[str, Any]:
        """
        Fetch model configuration from the API.
        Results are cached in-memory for CACHE_MODEL_CONFIG_TTL seconds.

        Args:
            config_id: The model configuration ID

        Returns:
            Model configuration dictionary

        Raises:
            ValueError: If API request fails or JWT token is missing
        """
        cached = await self._cache.get(config_id)
        if cached is not None:
            return cached

        if not settings.INTERNAL_API_KEY:
            raise ValueError("INTERNAL_API_KEY environment variable is not set")

        url = f"{settings.API_BASE_URL}/internal/model-configurations/{config_id}"
        headers = {
            "Authorization": f"ApiKey {settings.INTERNAL_API_KEY}",
            "Content-Type": "application/json"
        }
        timeout = httpx.Timeout(settings.API_TIMEOUT_SECONDS)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                await self._cache.set(config_id, data)
                return data
            except httpx.HTTPStatusError as e:
                from app.core.error_handling import log_api_error
                log_api_error(
                    error=e,
                    service_name="ModelConfigClient",
                    context="fetch_model_config()",
                    additional_info={"config_id": config_id, "status_code": e.response.status_code}
                )
                raise ValueError(f"Failed to fetch model config: HTTP {e.response.status_code}")
            except httpx.RequestError as e:
                from app.core.error_handling import log_api_error
                log_api_error(
                    error=e,
                    service_name="ModelConfigClient",
                    context="fetch_model_config()",
                    additional_info={"config_id": config_id}
                )
                err_msg = str(e).strip() or type(e).__name__
                raise ValueError(f"Failed to fetch model config: {err_msg}")

