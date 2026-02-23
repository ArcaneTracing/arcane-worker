"""
ContextEntityRecall evaluation using RAGAS.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import ContextEntityRecall
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ContextEntityRecallInput(BaseModel):
    """Input type for ContextEntityRecall evaluation."""
    reference: str = Field(..., description="Reference string")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    
    @field_validator('retrieved_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_context_entity_recall(
    input_data: ContextEntityRecallInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate ContextEntityRecall metric.
    
    Args:
        input_data: ContextEntityRecallInput containing reference and retrieved_contexts
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for ContextEntityRecall metric")
    
    # Create metric instance
    scorer = ContextEntityRecall(llm=llm)
    
    # Evaluate single input
    result = await scorer.ascore(
        reference=input_data.reference,
        retrieved_contexts=input_data.retrieved_contexts
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ContextEntityRecall",
        id=METRIC_IDS["context_entity_recall"]
    )

