"""
Unit tests for PromptVersionClient.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from app.clients.prompt_version_client import PromptVersionClient


class TestPromptVersionClient:
    """Tests for PromptVersionClient."""

    @pytest.fixture
    def client(self):
        """Client with mocked settings for tests."""
        with patch("app.clients.prompt_version_client.settings") as mock_settings:
            mock_settings.CACHE_PROMPT_VERSION_TTL = 60
            mock_settings.INTERNAL_API_KEY = "test-jwt-token"
            mock_settings.API_BASE_URL = "https://api.test"
            mock_settings.API_TIMEOUT_SECONDS = 30
            yield PromptVersionClient()

    @pytest.mark.asyncio
    async def test_fetch_latest_version_returns_cached_when_available(self, client):
        """Should return cached data when prompt_id is in cache."""
        cached_data = {"id": "pv-1", "promptId": "prompt-1", "modelConfigurationId": "mc-1"}
        client._cache.get = AsyncMock(return_value=cached_data)

        result = await client.fetch_latest_version("prompt-1")

        assert result == cached_data
        client._cache.get.assert_called_once_with("prompt-1")

    @pytest.mark.asyncio
    async def test_fetch_latest_version_raises_when_jwt_missing(self, client):
        """Should raise ValueError when INTERNAL_API_KEY is not set."""
        with patch("app.clients.prompt_version_client.settings") as mock_settings:
            mock_settings.CACHE_PROMPT_VERSION_TTL = 60
            mock_settings.INTERNAL_API_KEY = None
            mock_settings.API_BASE_URL = "https://api.test"
            mock_settings.API_TIMEOUT_SECONDS = 30
            c = PromptVersionClient()
            c._cache.get = AsyncMock(return_value=None)
            with pytest.raises(ValueError, match="INTERNAL_API_KEY environment variable is not set"):
                await c.fetch_latest_version("prompt-1")

    @pytest.mark.asyncio
    async def test_fetch_latest_version_calls_api_and_caches(self, client):
        """Should call API and cache result when not in cache."""
        client._cache.get = AsyncMock(return_value=None)
        client._cache.set = AsyncMock()
        api_response = {"id": "pv-1", "promptId": "prompt-1", "modelConfigurationId": "mc-1"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.json.return_value = api_response
            mock_response.raise_for_status = MagicMock()
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await client.fetch_latest_version("prompt-1")

            assert result == api_response
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert "prompt-1" in str(call_args)
            assert call_args[1]["headers"]["Authorization"] == "ApiKey test-jwt-token"
            client._cache.set.assert_called_once_with("prompt-1", api_response)

    @pytest.mark.asyncio
    async def test_fetch_latest_version_raises_on_http_error(self, client):
        """Should raise ValueError on HTTP status error."""
        client._cache.get = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Failed to fetch prompt version: HTTP 404"):
                await client.fetch_latest_version("prompt-1")
