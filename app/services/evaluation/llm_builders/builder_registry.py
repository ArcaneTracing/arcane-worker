"""
Registry for LLM builder factories.
Maps adapter types to builder factory functions.
"""
from app.core.instructor_genai_async_patch import apply_instructor_genai_async_patch

apply_instructor_genai_async_patch()

from typing import Dict, Any
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder
from app.core.registry import FactoryRegistry

# Import all builder classes
from app.services.evaluation.llm_builders.openai_llm_builder import OpenAILLMBuilder
from app.services.evaluation.llm_builders.azure_llm_builder import AzureLLMBuilder
from app.services.evaluation.llm_builders.anthropic_llm_builder import AnthropicLLMBuilder
from app.services.evaluation.llm_builders.bedrock_llm_builder import BedrockLLMBuilder
from app.services.evaluation.llm_builders.google_vertex_ai_llm_builder import GoogleVertexAILLMBuilder
from app.services.evaluation.llm_builders.google_ai_studio_llm_builder import GoogleAIStudioLLMBuilder


# Global builder registry
_builder_registry = FactoryRegistry[BaseLLMBuilder]()


def register_all_builders():
    """Register all LLM builders with the registry."""
    _builder_registry.register("anthropic", lambda config, config_dict: AnthropicLLMBuilder(config, config_dict))
    _builder_registry.register("openai", lambda config, config_dict: OpenAILLMBuilder(config, config_dict))
    _builder_registry.register("azure", lambda config, config_dict: AzureLLMBuilder(config, config_dict))
    _builder_registry.register("bedrock", lambda config, config_dict: BedrockLLMBuilder(config, config_dict))
    _builder_registry.register("google-vertex-ai", lambda config, config_dict: GoogleVertexAILLMBuilder(config, config_dict))
    _builder_registry.register("google-ai-studio", lambda config, config_dict: GoogleAIStudioLLMBuilder(config, config_dict))


def get_builder_registry() -> FactoryRegistry[BaseLLMBuilder]:
    """
    Get the global builder registry.
    
    Returns:
        The global FactoryRegistry instance
    """
    return _builder_registry


# Auto-register all builders when module is imported
register_all_builders()

