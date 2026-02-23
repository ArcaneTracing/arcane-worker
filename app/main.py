"""
Main FastStream application with FastAPI integration.
Combines RabbitMQ consumers and HTTP API endpoints.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import (
    EvaluationJobDto,
    ExperimentJobDto,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faststream.rabbit.schemas.queue import RabbitQueue
from app.config import settings
from app.core.error_handling import log_api_error
from app.core.logging_config import setup_logging
from app.domain.experiment.processor import ExperimentJobProcessor
from app.domain.evaluation.processor import EvaluationJobProcessor
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.core.broker_factory import get_broker

# Initialize logging
setup_logging()

logger = logging.getLogger(__name__)

async def _publish_to_dlq(
    payload: dict[str, Any],
    dlq_queue: str,
    error: Exception,
) -> None:
    """Publish failed message to dead letter queue for later processing."""
    dlq_message = {
        "payload": payload,
        "error": str(error),
    }
    try:
        await router.broker.publish(
            dlq_message,
            dlq_queue,
        )
        logger.info(f"Published failed job to DLQ {dlq_queue}")
    except Exception as dlq_error:
        logger.error(
            f"Failed to publish to DLQ {dlq_queue}: {dlq_error}. "
            f"Original error: {error}",
            exc_info=True,
        )
        raise# Create RabbitRouter for FastAPI integration
router = get_broker()

# Initialize processors
experiment_processor = ExperimentJobProcessor()
evaluation_processor = EvaluationJobProcessor()

experiment_queue = RabbitQueue(settings.EXPERIMENT_JOBS_TOPIC, durable=True) if settings.MESSAGE_BROKER == "rabbitmq" else settings.EXPERIMENT_JOBS_TOPIC
evaluation_queue = RabbitQueue(settings.EVALUATION_JOBS_TOPIC, durable=True) if settings.MESSAGE_BROKER == "rabbitmq" else settings.EVALUATION_JOBS_TOPIC

# Register experiment job subscriber
@router.subscriber(experiment_queue)
async def process_experiment_job(
    message: ExperimentJobDto,
) -> None:
    """Process a single experiment job from the queue."""

    try:
        logger.info(f"Processing experiment job (message_id={message.message_id})")

        result = await experiment_processor.process(message)
        
        # Publish result using router's broker
        experiment_id = result.experiment_id
        dataset_row_id = result.dataset_row_id
        await router.broker.publish(
            result.model_dump(by_alias=True),
            settings.EXPERIMENT_RESULTS_TOPIC,
        )
        
        logger.info(
            f"Published result: experimentId={experiment_id}, datasetRowId={dataset_row_id}, message_id={result.message_id}"
        )
        logger.info(f"Completed experiment job (message_id={message.message_id})")
        
    except Exception as e:
        log_api_error(
            error=e,
            service_name="process_experiment_job",
            context="process_experiment_job()",
            additional_info={"message_id": message.message_id}
        )
        await _publish_to_dlq(
            payload=message.model_dump(by_alias=True),
            dlq_queue=settings.EXPERIMENT_DLQ_QUEUE,
            error=e,
        )# Register evaluation job subscriber
@router.subscriber(evaluation_queue)
async def process_evaluation_job(
    message: EvaluationJobDto,
) -> None:
    """Process a single evaluation job from the queue."""

    try:
        logger.info(f"Processing evaluation job (message_id={message.message_id})")

        result = await evaluation_processor.process(message)
        
        # Publish result using router's broker
        evaluation_id = result.evaluation_id
        score_id = result.score_id
        await router.broker.publish(
            result.model_dump(by_alias=True),
            settings.EVALUATION_RESULTS_TOPIC,
        )
        
        logger.info(
            f"Published evaluation result: evaluationId={evaluation_id}, scoreId={score_id}, message_id={result.message_id}"
        )
        logger.info(f"Completed evaluation job (message_id={message.message_id})")
        
    except Exception as e:
        log_api_error(
            error=e,
            service_name="process_evaluation_job",
            context="process_evaluation_job()",
            additional_info={"message_id": message.message_id}
        )
        await _publish_to_dlq(
            payload=message.model_dump(by_alias=True),
            dlq_queue=settings.EVALUATION_DLQ_QUEUE,
            error=e,
        )# Create FastAPI app
app = FastAPI(
    title="Arcane Eval Chat API",
    description="REST API for running chat prompts across different LLM models",
    version="1.0.0",
)

# Add CORS middleware to FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include FastStream router (handles both subscribers and can handle HTTP routes)
app.include_router(router)

# Register HTTP routes on FastAPI app
app.include_router(health_router)
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

if __name__ == "__main__":
    # Run FastAPI app (FastStream router is included and will handle message subscriptions)
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
