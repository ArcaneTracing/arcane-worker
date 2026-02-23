"""
Common parameter extraction utilities for LLM services.
Centralizes logic for extracting temperature, max_tokens, and other parameters.
"""
from typing import Dict, Any, Optional
from app.models.schemas import LLMServiceRequestDto


def extract_temperature_and_max_tokens(
    request: LLMServiceRequestDto,
    provider_type: Optional[str] = None
) -> tuple[float, int]:
    """
    Extract temperature and max_tokens from invocation parameters or model config.
    
    This is a common pattern used across multiple LLM services.
    
    Args:
        request: LLM service request DTO
        provider_type: Optional provider type to check in invocation_params (e.g., "openai", "anthropic", "azure_openai")
        
    Returns:
        Tuple of (temperature, max_tokens) with defaults applied
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters
    
    temperature = None
    max_tokens = None
    
    # Try to extract from provider-specific invocation params
    if provider_type and hasattr(invocation_params, provider_type):
        provider_params = getattr(invocation_params, provider_type)
        if isinstance(provider_params, dict):
            temperature = provider_params.get("temperature") or model_config.temperature
            max_tokens = (
                provider_params.get("max_tokens") or
                provider_params.get("max_completion_tokens") or
                model_config.max_tokens
            )
    
    # Fallback to model config defaults
    if temperature is None:
        temperature = model_config.temperature or 0.7
    if max_tokens is None:
        max_tokens = model_config.max_tokens or 1000
    
    return temperature, max_tokens


def extract_temperature_and_max_tokens_openai(request: LLMServiceRequestDto) -> tuple[float, int]:
    """
    Extract temperature and max_tokens for OpenAI/Azure services.
    
    Args:
        request: LLM service request DTO
        
    Returns:
        Tuple of (temperature, max_tokens)
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters
    
    if invocation_params.type == "openai":
        openai_params = invocation_params.openai
        temperature = openai_params.get("temperature") or model_config.temperature or 0.7
        max_tokens = (
            openai_params.get("max_tokens") or
            openai_params.get("max_completion_tokens") or
            model_config.max_tokens or
            1000
        )
    elif invocation_params.type == "azure_openai":
        azure_params = invocation_params.azure_openai
        temperature = azure_params.get("temperature") or model_config.temperature or 0.7
        max_tokens = (
            azure_params.get("max_tokens") or
            azure_params.get("max_completion_tokens") or
            model_config.max_tokens or
            1000
        )
    else:
        temperature = model_config.temperature or 0.7
        max_tokens = model_config.max_tokens or 1000
    
    return temperature, max_tokens


def extract_temperature_and_max_tokens_anthropic(request: LLMServiceRequestDto) -> tuple[float, int]:
    """
    Extract temperature and max_tokens for Anthropic services.
    
    Args:
        request: LLM service request DTO
        
    Returns:
        Tuple of (temperature, max_tokens)
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters
    
    if invocation_params.type == "anthropic":
        anthropic_params = invocation_params.anthropic
        temperature = anthropic_params.get("temperature") or model_config.temperature or 0.7
        max_tokens = anthropic_params.get("max_tokens") or model_config.max_tokens or 1000
    else:
        temperature = model_config.temperature or 0.7
        max_tokens = model_config.max_tokens or 1000
    
    return temperature, max_tokens


def extract_temperature_and_max_tokens_google_ai_studio(
    request: LLMServiceRequestDto,
) -> tuple[float, int]:
    """
    Extract temperature and max_tokens for Google AI Studio (Gemini API) services.
    Handles both "google-ai-studio" and "google" invocation parameter types.

    Args:
        request: LLM service request DTO

    Returns:
        Tuple of (temperature, max_tokens)
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters

    params = None
    if invocation_params.type == "google-ai-studio":
        params = getattr(invocation_params, "google_ai_studio", None)
    elif invocation_params.type == "google":
        params = getattr(invocation_params, "google", None)

    if isinstance(params, dict):
        temperature = (
            params.get("temperature")
            or model_config.temperature
            or 0.7
        )
        max_tokens = (
            params.get("max_tokens")
            or params.get("maxTokens")
            or model_config.max_tokens
            or 1000
        )
    else:
        temperature = model_config.temperature or 0.7
        max_tokens = model_config.max_tokens or 1000

    return temperature, max_tokens


def extract_temperature_and_max_tokens_google_vertex_ai(
    request: LLMServiceRequestDto,
) -> tuple[float, int]:
    """
    Extract temperature and max_tokens for Google Vertex AI services.
    Handles both "google-vertex-ai" and "google" invocation parameter types.

    Args:
        request: LLM service request DTO

    Returns:
        Tuple of (temperature, max_tokens)
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters

    params = None
    if invocation_params.type == "google-vertex-ai":
        params = getattr(invocation_params, "google_vertex_ai", None)
    elif invocation_params.type == "google":
        params = getattr(invocation_params, "google", None)

    if isinstance(params, dict):
        temperature = (
            params.get("temperature")
            or model_config.temperature
            or 0.7
        )
        max_tokens = (
            params.get("max_tokens")
            or params.get("maxTokens")
            or model_config.max_tokens
            or 1000
        )
    else:
        temperature = model_config.temperature or 0.7
        max_tokens = model_config.max_tokens or 1000

    return temperature, max_tokens


def extract_temperature_and_max_tokens_bedrock(
    request: LLMServiceRequestDto,
) -> tuple[float, int]:
    """
    Extract temperature and max_tokens for AWS Bedrock services.
    Handles "bedrock" and "aws" invocation parameter types.
    """
    model_config = request.model_configuration.configuration
    invocation_params = request.prompt_version.invocation_parameters

    params = None
    if hasattr(invocation_params, "type"):
        if invocation_params.type == "bedrock":
            params = getattr(invocation_params, "bedrock", None)
        elif invocation_params.type == "aws":
            params = getattr(invocation_params, "aws", None)

    if isinstance(params, dict):
        temperature = (
            params.get("temperature")
            or model_config.temperature
            or 0.7
        )
        max_tokens = (
            params.get("max_tokens")
            or params.get("maxTokens")
            or model_config.max_tokens
            or 1000
        )
    else:
        temperature = model_config.temperature or 0.7
        max_tokens = model_config.max_tokens or 1000

    return temperature, max_tokens

