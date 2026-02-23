"""
NvidiaAnswerAccuracy evaluation using RAGAS.
"""
import logging
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import AnswerAccuracy
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS

logger = logging.getLogger(__name__)

class NvidiaAnswerAccuracyInput(BaseModel):
    """Input type for NvidiaAnswerAccuracy evaluation."""
    user_input: str = Field(..., description="User input string")
    response: str = Field(..., description="Response string")
    reference: str = Field(..., description="Reference string")
    
    @field_validator('user_input', 'response', 'reference', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)


async def evaluate_nvidia_answer_accuracy(
    input_data: NvidiaAnswerAccuracyInput,
    llm: BaseRagasLLM
) -> RagasScore:
    """
    Evaluate NvidiaAnswerAccuracy metric.
    
    Args:
        input_data: NvidiaAnswerAccuracyInput containing user_input, response, and reference
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for AnswerAccuracy metric")
    
    # Create metric instance
    scorer = AnswerAccuracy(llm=llm)
    
    try:
        # Evaluate single input
        result = await scorer.ascore(
            user_input=input_data.user_input,
            response=input_data.response,
            reference=input_data.reference
        )
        # log the result
        logger.info(f"NvidiaAnswerAccuracy result: {result}")
        score = extract_score(result)
        
        return RagasScore(
            score=score,
            metric="NvidiaAnswerAccuracy",
            id=METRIC_IDS["nvidia_answer_accuracy"]
        )
    except Exception as e:
        from app.core.error_handling import log_api_error
        log_api_error(
            error=e,
            service_name="NvidiaAnswerAccuracy",
            context="evaluate_nvidia_answer_accuracy()"
        )
        raise

