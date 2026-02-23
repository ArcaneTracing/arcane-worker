"""
ContextUtilisation evaluation using RAGAS.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import ContextUtilization
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string, convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ContextUtilisationInput(BaseModel):
    """Input type for ContextUtilisation evaluation."""
    user_input: str = Field(..., description="User input string")
    response: str = Field(..., description="Response string")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    
    @field_validator('user_input', 'response', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)
    
    @field_validator('retrieved_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_context_utilisation(
    input_data: ContextUtilisationInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate ContextUtilisation metric.
    
    Args:
        input_data: ContextUtilisationInput containing user_input, response, and retrieved_contexts
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for ContextUtilization metric")
    
    # Create metric instance
    scorer = ContextUtilization(llm=llm)
    
    # Evaluate single input
    result = await scorer.ascore(
        user_input=input_data.user_input,
        response=input_data.response,
        retrieved_contexts=input_data.retrieved_contexts
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ContextUtilisation",
        id=METRIC_IDS["context_utilisation"]
    )

