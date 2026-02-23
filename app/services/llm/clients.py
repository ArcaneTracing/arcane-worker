"""
Shared client factories for LLM providers.
These clients can be used by both LLM services and Ragas builders.
"""
import logging
from typing import Any, Optional
from openai import OpenAI, AzureOpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic

logger = logging.getLogger(__name__)


def _patch_anthropic_client_filter_top_p(client: AsyncAnthropic) -> None:
    """
    Monkey-patch client.messages.create to filter top_p from kwargs.
    Anthropic rejects requests with both temperature and top_p.
    RAGAS adds top_p=0.1 by default via InstructorModelArgs.
    Must use monkey-patch (not a wrapper) so instructor.from_anthropic's
    isinstance(client, AsyncAnthropic) check passes.
    """
    original_create = client.messages.create

    async def filtered_create(**kwargs: Any) -> Any:
        kwargs.pop("top_p", None)
        return await original_create(**kwargs)

    client.messages.create = filtered_create  # type: ignore[method-assign]


def create_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None
) -> OpenAI:
    """
    Create an OpenAI client instance.
    
    Args:
        api_key: OpenAI API key
        base_url: Optional custom base URL
        organization: Optional organization ID
        
    Returns:
        OpenAI client instance
    """
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    
    return OpenAI(**kwargs)


def create_azure_openai_client(
    api_key: str,
    endpoint: str,
    api_version: str = "2024-02-15-preview",
    deployment_name: Optional[str] = None,
    use_azure_class: bool = False
) -> OpenAI | AzureOpenAI:
    """
    Create an Azure OpenAI client instance.
    
    Args:
        api_key: Azure OpenAI API key
        endpoint: Azure endpoint (e.g., https://{resource}.openai.azure.com)
        api_version: API version
        deployment_name: Optional deployment name
        use_azure_class: If True, use AzureOpenAI class; if False, use OpenAI with custom base_url
        
    Returns:
        OpenAI or AzureOpenAI client instance configured for Azure
    """
    if use_azure_class:
        # Use AzureOpenAI class (used by services)
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    else:
        # Use OpenAI with custom base_url (used by builders for Ragas)
        if deployment_name:
            base_url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}"
        else:
            base_url = endpoint.rstrip('/')
        
        kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "default_query": {"api-version": api_version}
        }
        
        return OpenAI(**kwargs)


def create_anthropic_client(
    api_key: str,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None
) -> Anthropic:
    """
    Create an Anthropic client instance.
    
    Args:
        api_key: Anthropic API key
        base_url: Optional custom base URL
        timeout: Optional timeout in seconds
        
    Returns:
        Anthropic client instance
    """
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout:
        kwargs["timeout"] = timeout
    
    return Anthropic(**kwargs)


def create_async_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None
) -> AsyncOpenAI:
    """
    Create an async OpenAI client instance for use with RAGAS.
    
    Args:
        api_key: OpenAI API key
        base_url: Optional custom base URL
        organization: Optional organization ID
        
    Returns:
        AsyncOpenAI client instance
    """
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    
    return AsyncOpenAI(**kwargs)


def create_async_azure_openai_client(
    api_key: str,
    endpoint: str,
    api_version: str = "2024-02-15-preview",
    deployment_name: Optional[str] = None
) -> AsyncOpenAI:
    """
    Create an async Azure OpenAI client instance for use with RAGAS.
    
    Args:
        api_key: Azure OpenAI API key
        endpoint: Azure endpoint (e.g., https://{resource}.openai.azure.com)
        api_version: API version
        deployment_name: Optional deployment name
        
    Returns:
        AsyncOpenAI client instance configured for Azure
    """
    if deployment_name:
        base_url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}"
    else:
        base_url = endpoint.rstrip('/')
    
    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "default_query": {"api-version": api_version}
    }
    
    return AsyncOpenAI(**kwargs)


def create_async_anthropic_client(
    api_key: str,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None
) -> AsyncAnthropic:
    """
    Create an async Anthropic client instance for use with RAGAS.
    Patches client.messages.create to filter out top_p (Anthropic rejects
    both temperature and top_p; RAGAS adds top_p=0.1 by default).
    
    Args:
        api_key: Anthropic API key
        base_url: Optional custom base URL
        timeout: Optional timeout in seconds
        
    Returns:
        AsyncAnthropic client instance (patched to filter top_p)
    """
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout:
        kwargs["timeout"] = timeout

    client = AsyncAnthropic(**kwargs)
    _patch_anthropic_client_filter_top_p(client)
    return client

