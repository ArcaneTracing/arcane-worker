"""
Registry for LLM model services.
Maps adapter types to service factory functions.
"""
from typing import Dict, Any
from app.models.schemas import BaseModelConfiguration, AdapterType
from app.services.llm.base import BaseModelService
from app.core.registry import FactoryRegistry

# Import all service classes
from app.services.llm.openai_service import OpenAIModelService
from app.services.llm.azure_service import AzureModelService
from app.services.llm.anthropic_service import AnthropicModelService
from app.services.llm.bedrock_service import BedrockModelService
from app.services.llm.google_ai_studio_service import GoogleAIStudioModelService
from app.services.llm.google_vertex_ai_service import GoogleVertexAIModelService


# Global service registry
_service_registry = FactoryRegistry[BaseModelService]()


def _create_openai_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for OpenAI service."""
    return OpenAIModelService(api_key=api_key)


def _create_azure_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for Azure OpenAI service."""
    azure_config = model_config.config or {}
    endpoint = azure_config.get("endpoint", "")
    if not endpoint:
        raise ValueError("Azure endpoint is required in config")
    api_version = azure_config.get("api_version", "2024-02-15-preview")
    return AzureModelService(
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version
    )


def _create_anthropic_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for Anthropic service."""
    return AnthropicModelService(api_key=api_key)


def _create_google_ai_studio_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for Google AI Studio (Gemini API) service."""
    return GoogleAIStudioModelService(api_key=api_key)


def _create_google_vertex_ai_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for Google Vertex AI (Gemini API) service.

    Requires api_key to contain service account credentials JSON (paste from GCP Console).
    Optional config.project and config.location for project/location.
    """
    vertex_config = model_config.config or {}
    project = vertex_config.get("project")
    location = vertex_config.get("location")
    return GoogleVertexAIModelService(
        credentials_json=api_key,
        project=project,
        location=location,
    )


def _create_bedrock_service(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
    """Factory function for AWS Bedrock service.

    Requires config.region. Optional: config.endpointUrl, config.awsAccessKeyId, config.awsSecretAccessKey.
    Uses default credential chain if AWS keys not provided.
    """
    bedrock_config = model_config.config or {}
    region = bedrock_config.get("region")
    if not region:
        raise ValueError("Bedrock requires config.region")
    return BedrockModelService(
        region=region,
        endpoint_url=bedrock_config.get("endpointUrl"),
        aws_access_key_id=bedrock_config.get("awsAccessKeyId"),
        aws_secret_access_key=bedrock_config.get("awsSecretAccessKey"),
    )


def _create_not_implemented_service(adapter: str):
    """Factory function for not-yet-implemented services."""
    def factory(model_config: BaseModelConfiguration, api_key: str) -> BaseModelService:
        raise ValueError(f"Adapter type {adapter} is not yet implemented")
    return factory


def register_all_services():
    """Register all LLM model services with the registry."""
    _service_registry.register(AdapterType.OPENAI.value, _create_openai_service)
    _service_registry.register(AdapterType.AZURE.value, _create_azure_service)
    _service_registry.register(AdapterType.ANTHROPIC.value, _create_anthropic_service)
    _service_registry.register(AdapterType.BEDROCK.value, _create_bedrock_service)
    _service_registry.register(AdapterType.GOOGLE_VERTEX_AI.value, _create_google_vertex_ai_service)
    _service_registry.register(AdapterType.GOOGLE_AI_STUDIO.value, _create_google_ai_studio_service)


def get_service_registry() -> FactoryRegistry[BaseModelService]:
    """
    Get the global service registry.
    
    Returns:
        The global FactoryRegistry instance
    """
    return _service_registry


# Auto-register all services when module is imported
register_all_services()

