"""
Processes evaluation jobs.
"""
from __future__ import annotations

import logging

from app.clients.model_config_client import ModelConfigClient
from app.clients.prompt_version_client import PromptVersionClient
from app.models.schemas import (
    EvaluationJobDto,
    EvaluationResultDto,
    ModelConfigurationWithEncryptedKey,
)
from app.services.evaluation.llm_evaluation_processor import LLMEvaluationProcessor
from app.services.evaluation.ragas.ragas_processor import RagasProcessor

logger = logging.getLogger(__name__)


class EvaluationJobProcessor:
    """Processes individual evaluation jobs"""

    def __init__(self) -> None:
        """Initialize the processor with required processors."""
        self.ragas_processor = RagasProcessor()
        self.llm_evaluation_processor = LLMEvaluationProcessor()
        self.model_config_client = ModelConfigClient()
        self.prompt_version_client = PromptVersionClient()
    
    async def process(self, job_data: EvaluationJobDto) -> EvaluationResultDto:
        """
        Process a single evaluation job.
        
        Args:
            job_data: Job data DTO containing evaluationId, scoreId, scoringType,
                datasetRowId, experimentResultId, and RAGAS or LLM-specific fields.
            
        Returns:
            Result DTO to be published
        """
        try:
            logger.info(
                f"Starting to process evaluation job: evaluationId={job_data.evaluation_id}, "
                f"scoreId={job_data.score_id}"
            )
            
            scoring_type = job_data.scoring_type
            score_mapping = job_data.score_mapping or {}
            
            logger.info(
                f"Processing evaluation job: evaluationId={job_data.evaluation_id}, "
                f"ragasModelConfigurationId={job_data.ragas_model_configuration_id}, "
                f"scoreId={job_data.score_id}, ragasScoreKey={job_data.ragas_score_key}"
            )
            
            # Handle non-RAGAS evaluations using LLMEvaluationProcessor
            if scoring_type != "RAGAS":
                logger.info(f"Processing non-RAGAS evaluation job with scoringType={scoring_type}")

                prompt_id = job_data.prompt_id

                if not prompt_id:
                    raise ValueError("promptId is required for non-RAGAS evaluation job")
                if not score_mapping:
                    raise ValueError("scoreMapping is required for non-RAGAS evaluation job")

                # Fetch latest prompt version from API
                prompt_version_dict = await self.prompt_version_client.fetch_latest_version(
                    prompt_id,
                )

                # Fetch model configuration from API (from prompt version)
                model_configuration_id = prompt_version_dict.get("modelConfigurationId")
                if not model_configuration_id:
                    raise ValueError(
                        "Prompt version must have modelConfigurationId"
                    )
                config_dict = await self.model_config_client.fetch_model_config(
                    model_configuration_id
                )
                model_configuration = ModelConfigurationWithEncryptedKey(**config_dict)

                score = await self.llm_evaluation_processor.evaluate(
                    model_configuration=model_configuration.model_dump(by_alias=True),
                    prompt_version=prompt_version_dict,
                    inputs=score_mapping,
                )
                
                logger.info(f"Non-RAGAS evaluation completed. Score: {score}")
                
                return EvaluationResultDto(
                    evaluation_id=job_data.evaluation_id,
                    score_id=job_data.score_id,
                    dataset_row_id=job_data.dataset_row_id,
                    experiment_result_id=job_data.experiment_result_id,
                    score=str(score),
                    message_id=job_data.message_id,
                )
            
            ragas_score_key = job_data.ragas_score_key
            if not ragas_score_key:
                raise ValueError("ragasScoreKey is required for evaluation job")
            
            logger.info(f"Processing evaluation with ragasScoreKey: {ragas_score_key}")
            result = await self.ragas_processor.evaluate(
                ragas_score_key=ragas_score_key,
                score_mapping=score_mapping,
                ragas_model_config_id=job_data.ragas_model_configuration_id
            )
            
            logger.info(f"Evaluation completed. Result: score={result.score}, metric={result.metric}")
            
            return EvaluationResultDto(
                evaluation_id=job_data.evaluation_id,
                score_id=job_data.score_id,
                dataset_row_id=job_data.dataset_row_id,
                experiment_result_id=job_data.experiment_result_id,
                score=str(result.score),
                metric=result.metric,
                metric_id=result.id,
                message_id=job_data.message_id,
            )
            
        except Exception as e:
            from app.core.error_handling import log_api_error
            log_api_error(
                error=e,
                service_name="EvaluationJobProcessor",
                context="process()",
                additional_info={
                    "evaluationId": job_data.evaluation_id,
                    "scoreId": job_data.score_id
                }
            )
            return EvaluationResultDto(
                evaluation_id=job_data.evaluation_id,
                score_id=job_data.score_id,
                dataset_row_id=job_data.dataset_row_id,
                experiment_result_id=job_data.experiment_result_id,
                score=None,
                error=str(e),
                message_id=job_data.message_id,
            )

