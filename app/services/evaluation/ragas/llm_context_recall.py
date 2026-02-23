"""
LLMContextRecall evaluation using RAGAS.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import ContextRecall
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string, convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class LLMContextRecallInput(BaseModel):
    """Input type for LLMContextRecall evaluation."""
    user_input: str = Field(..., description="User input string")
    reference: str = Field(..., description="Reference string")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    
    @field_validator('user_input', 'reference', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)
    
    @field_validator('retrieved_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_llm_context_recall(
    input_data: LLMContextRecallInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate LLMContextRecall metric.
    
    Args:
        input_data: LLMContextRecallInput containing user_input, reference, and retrieved_contexts
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for ContextRecall metric")
    
    # Create metric instance
    scorer = ContextRecall(llm=llm)
    
    # Evaluate single input
    result = await scorer.ascore(
        user_input=input_data.user_input,
        retrieved_contexts=input_data.retrieved_contexts,
        reference=input_data.reference
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="LLMContextRecall",
        id=METRIC_IDS["llm_context_recall"]
    )

