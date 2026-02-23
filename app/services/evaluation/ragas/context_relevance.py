"""
ContextRelevance evaluation using RAGAS.
"""
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import ContextRelevance
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string, convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS

logger = logging.getLogger(__name__)

class ContextRelevanceInput(BaseModel):
    """Input type for ContextRelevance evaluation."""
    user_input: str = Field(..., description="User input string")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    
    @field_validator('user_input', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)
    
    @field_validator('retrieved_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_context_relevance(
    input_data: ContextRelevanceInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate ContextRelevance metric.
    
    Args:
        input_data: ContextRelevanceInput containing user_input and retrieved_contexts
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for ContextRelevance metric")
    
    # Create metric instance
    scorer = ContextRelevance(llm=llm)
    
    logger.info(f"Evaluating ContextRelevance metric with input: {input_data}")
    
    # Evaluate single input
    result = await scorer.ascore(
        user_input=input_data.user_input,
        retrieved_contexts=input_data.retrieved_contexts
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ContextRelevance",
        id=METRIC_IDS["context_relevance"]
    )

