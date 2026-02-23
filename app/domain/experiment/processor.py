"""
Processes experiment jobs.
"""
from __future__ import annotations

import logging
import time

from app.clients.model_config_client import ModelConfigClient
from app.clients.prompt_version_client import PromptVersionClient
from app.models.schemas import (
    ExperimentJobDto,
    ExperimentResultDto,
    LLMServiceRequestDto,
    ModelConfigurationWithEncryptedKey,
    PromptVersionDto,
)
from app.services.llm.service import ModelService

logger = logging.getLogger(__name__)


class ExperimentJobProcessor:
    """Processes individual experiment jobs"""

    def __init__(self) -> None:
        self.model_service = ModelService()
        self.model_config_client = ModelConfigClient()
        self.prompt_version_client = PromptVersionClient()
    
    async def process(self, job_data: ExperimentJobDto) -> ExperimentResultDto:
        """
        Process a single experiment job.
        
        Args:
            job_data: Job data DTO from the queue
            
        Returns:
            Result DTO to be published
        """
        start_time = time.time()
        experiment_job = job_data
        
        try:
            
            logger.info(
                f"Processing experiment job: experimentId={experiment_job.experiment_id}, "
                f"datasetRowId={experiment_job.dataset_row_id}"
            )

            # Fetch latest prompt version from API
            prompt_version_dict = await self.prompt_version_client.fetch_latest_version(
                experiment_job.prompt_id,
            )
            prompt_version = PromptVersionDto.model_validate(prompt_version_dict)

            # Fetch model configuration from API (from prompt version)
            config_dict = await self.model_config_client.fetch_model_config(
                prompt_version.model_configuration_id
            )
            model_configuration = ModelConfigurationWithEncryptedKey(**config_dict)

            # Convert to LLMServiceRequestDto format
            llm_request = LLMServiceRequestDto(
                model_configuration=model_configuration,
                prompt_version=prompt_version,
                inputs=experiment_job.inputs,
            )
            
            # Execute LLM request
            response = await self.model_service.execute(llm_request)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract metadata
            metadata = {
                "execution_time_ms": execution_time_ms
            }
            
            if response.get("usage"):
                usage = response["usage"]
                if isinstance(usage, dict):
                    metadata["tokens_used"] = usage.get("total_tokens", 0)
            
            # Build result
            result = ExperimentResultDto(
                experiment_id=experiment_job.experiment_id,
                dataset_row_id=experiment_job.dataset_row_id,
                result=response.get("output", ""),
                metadata=metadata,
                message_id=experiment_job.message_id,
            )
            
            logger.info(
                f"Successfully processed job: experimentId={experiment_job.experiment_id}, "
                f"datasetRowId={experiment_job.dataset_row_id}, "
                f"execution_time_ms={execution_time_ms}"
            )
            
            return result
            
        except Exception as e:
            from app.core.error_handling import log_api_error
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            log_api_error(
                error=e,
                service_name="ExperimentJobProcessor",
                context="process()"
            )
            
            return ExperimentResultDto(
                experiment_id=experiment_job.experiment_id,
                dataset_row_id=experiment_job.dataset_row_id,
                result="",
                metadata={"execution_time_ms": execution_time_ms},
                error=str(e),
                message_id=experiment_job.message_id,
            )

