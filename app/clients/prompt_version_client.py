"""
Client for fetching prompt versions from the API.

Uses in-memory TTL cache to avoid redundant API calls when processing
many jobs with the same prompt.
"""
import logging
from typing import Dict, Any
import httpx
from app.config import settings
from app.core.cache import TTLCache

logger = logging.getLogger(__name__)


class PromptVersionClient:
    """Client for fetching prompt versions from the internal API."""

    def __init__(self) -> None:
        self._cache = TTLCache(ttl_seconds=settings.CACHE_PROMPT_VERSION_TTL)

    async def fetch_latest_version(self, prompt_id: str) -> Dict[str, Any]:
        """
        Fetch the latest prompt version for a prompt.

        Args:
            prompt_id: The prompt ID (UUID) - globally unique

        Returns:
            Prompt version dictionary (PromptVersionDto shape)

        Raises:
            ValueError: If API request fails or JWT token is missing
        """
        cached = await self._cache.get(prompt_id)
        if cached is not None:
            return cached

        if not settings.INTERNAL_API_KEY:
            raise ValueError("INTERNAL_API_KEY environment variable is not set")

        url = f"{settings.API_BASE_URL}/internal/prompts/{prompt_id}/latest-version"
        headers = {
            "Authorization": f"ApiKey {settings.INTERNAL_API_KEY}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(settings.API_TIMEOUT_SECONDS)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                await self._cache.set(prompt_id, data)
                return data
            except httpx.HTTPStatusError as e:
                from app.core.error_handling import log_api_error

                log_api_error(
                    error=e,
                    service_name="PromptVersionClient",
                    context="fetch_latest_version()",
                    additional_info={
                        "prompt_id": prompt_id,
                        "status_code": e.response.status_code,
                    },
                )
                raise ValueError(
                    f"Failed to fetch prompt version: HTTP {e.response.status_code}"
                )
            except httpx.RequestError as e:
                from app.core.error_handling import log_api_error

                log_api_error(
                    error=e,
                    service_name="PromptVersionClient",
                    context="fetch_latest_version()",
                    additional_info={"prompt_id": prompt_id},
                )
                err_msg = str(e).strip() or type(e).__name__
                raise ValueError(f"Failed to fetch prompt version: {err_msg}")
