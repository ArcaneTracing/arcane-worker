"""
Processor for LLM-based evaluation jobs (non-RAGAS).
"""
import logging
from typing import Dict, Any
from app.models.schemas import LLMServiceRequestDto
from app.services.llm.service import ModelService

logger = logging.getLogger(__name__)


class LLMEvaluationProcessor:
    """Processor for LLM-based evaluation jobs"""
    
    def __init__(self):
        """Initialize the processor with ModelService."""
        self.model_service = ModelService()
    
    async def evaluate(
        self,
        model_configuration: Dict[str, Any],
        prompt_version: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> str:
        """
        Execute LLM evaluation request.
        
        Args:
            model_configuration: Model configuration dictionary
            prompt_version: Prompt version dictionary
            inputs: Input variables for template rendering
            
        Returns:
            Score as string (extracted from LLM output)
        """
        logger.info("Processing LLM evaluation request")
        
        try:
            # Convert dictionaries to DTOs
            llm_request = LLMServiceRequestDto(
                model_configuration=model_configuration,
                prompt_version=prompt_version,
                inputs=inputs
            )
            
            # Execute LLM request
            response = await self.model_service.execute(llm_request)
            
            # Extract score from output
            score = response.get("output", "")
            
            logger.info(f"LLM evaluation completed. Score: {score}")
            
            return score
        
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="LLMEvaluationProcessor",
                context="evaluate()"
            )
            raise

