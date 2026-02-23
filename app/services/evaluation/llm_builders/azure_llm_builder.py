"""
Builder for Azure OpenAI models.
"""
import logging
from typing import TYPE_CHECKING, Optional
from openai import AsyncOpenAI
from app.services.evaluation.llm_builders.base_llm_builder import BaseLLMBuilder
from app.services.llm.clients import create_async_azure_openai_client

logger = logging.getLogger(__name__)


class AzureLLMBuilder(BaseLLMBuilder):
    """Builder for Azure OpenAI models"""
    
    def build_client(self) -> AsyncOpenAI:
        """Build async Azure OpenAI client instance for RAGAS"""
        logger.debug(f"Building async Azure OpenAI client for model: {self.model_name}")
        endpoint = self.config_dict.get("endpoint")
        if not endpoint:
            error_msg = "Azure endpoint is required in config"
            logger.error(f"{error_msg}. Config dict keys: {list(self.config_dict.keys())}")
            raise ValueError(error_msg)
        
        api_version = self.config_dict.get("apiVersion", "2024-02-15-preview")
        deployment_name = self.config_dict.get("deploymentName", self.model_name)
        
        logger.debug(f"Azure endpoint: {endpoint}, Deployment: {deployment_name}, API version: {api_version}")
        
        return create_async_azure_openai_client(
            api_key=self.api_key,
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name
        )
    
    def get_provider(self) -> Optional[str]:
        """Get provider name for Azure OpenAI"""
        return "openai"

