"""
RAGAS processor that routes evaluation requests to the appropriate evaluation function.
Uses Strategy pattern with metric registry for routing.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.evaluation.ragas.metric_registry import MetricRegistry
from app.clients.model_config_client import ModelConfigClient
from app.services.evaluation.llm_builders.llm_builder import LLMBuilder
from app.services.evaluation.embeddings_builder import EmbeddingsBuilder
from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.metric_registry import get_registry
from app.services.evaluation.ragas.json_parser import parse_json_strings

# Import metrics registration to ensure all metrics are registered
from app.services.evaluation.ragas import metrics_registration  # noqa: F401

logger = logging.getLogger(__name__)


class RagasProcessor:
    """
    Processor for routing RAGAS evaluation requests to appropriate functions.
    
    Uses Strategy pattern with metric registry for dynamic routing.
    """
    
    def __init__(
        self,
        model_config_client: Optional[ModelConfigClient] = None,
        llm_builder: Optional[LLMBuilder] = None,
        embeddings_builder: Optional[EmbeddingsBuilder] = None,
        registry: Optional[MetricRegistry] = None
    ) -> None:
        """
        Initialize the processor with required clients and builders.
        
        Args:
            model_config_client: Client for fetching model configurations (optional, creates default)
            llm_builder: Builder for creating LLM instances (optional, creates default)
            embeddings_builder: Builder for creating embeddings instances (optional, creates default)
            registry: Metric registry instance (optional, uses global registry)
        """
        self.model_config_client = model_config_client or ModelConfigClient()
        self.llm_builder = llm_builder or LLMBuilder()
        self.embeddings_builder = embeddings_builder or EmbeddingsBuilder()
        self.registry = registry or get_registry()
    
    async def evaluate(
        self,
        ragas_score_key: str,
        score_mapping: Dict[str, Any],
        ragas_model_config_id: str,
    ) -> RagasScore:
        """
        Route evaluation request to the appropriate RAGAS evaluation function.
        
        Uses the metric registry to find and execute the appropriate metric.
        
        Args:
            ragas_score_key: The metric ID (UUID) for the evaluation function
            score_mapping: Dictionary that matches perfectly the input class expected by the function
            ragas_model_config_id: The model configuration ID to fetch and use for LLM creation
            
        Returns:
            RagasScore containing evaluation results
            
        Raises:
            ValueError: If ragas_score_key is unknown or score_mapping is invalid
        """
        logger.info(f"Routing evaluation request for ragasScoreKey: {ragas_score_key}")
        
        # Get metric from registry
        metric = self.registry.get(ragas_score_key)
        if not metric:
            raise ValueError(f"Unknown ragasScoreKey: {ragas_score_key}")
        
        logger.info(f"Found metric: {metric.metric_name} (requires_llm={metric.requires_llm}, requires_embeddings={metric.requires_embeddings})")
        
        # Fetch model configuration from API (only if needed)
        model_config = None
        llm_instance = None
        embeddings_instance = None
        
        if metric.requires_llm or metric.requires_embeddings:
            logger.info(f"Fetching model config for ID: {ragas_model_config_id}")
            model_config = await self.model_config_client.fetch_model_config(ragas_model_config_id)
            logger.info(f"Fetched model config for {ragas_model_config_id}")
        
        # Create LLM instance if required
        if metric.requires_llm:
            logger.info("Creating LLM instance from model config")
            llm_instance = self.llm_builder.build_from_config(model_config)
            logger.info("Successfully created LLM instance")
        
        # Create embeddings instance if required
        if metric.requires_embeddings:
            logger.info("Creating embeddings instance from model config")
            embeddings_instance = self.embeddings_builder.build_from_config(model_config)
            logger.info("Successfully created embeddings instance")
        
        # Parse JSON strings in score_mapping to handle cases where values are JSON strings
        # instead of proper Python types (e.g., '["item"]' instead of ["item"])
        logger.info(f"Parsing JSON strings in score_mapping: {score_mapping}")
        parsed_score_mapping = parse_json_strings(score_mapping)
        
        # Validate and parse input data
        input_data = metric.validate_input(parsed_score_mapping)
        
        # Execute the metric evaluation
        return await metric.evaluate(
            input_data=input_data,
            llm=llm_instance,
            embeddings=embeddings_instance
        )

