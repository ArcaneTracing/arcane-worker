"""
NonLLMContextRecall evaluation using RAGAS.
"""
from typing import List
from pydantic import BaseModel, Field, field_validator
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class NonLLMContextRecallInput(BaseModel):
    """Input type for NonLLMContextRecall evaluation."""
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    reference_contexts: List[str] = Field(..., description="List of reference context strings")
    
    @field_validator('retrieved_contexts', 'reference_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_non_llm_context_recall(
    input_data: NonLLMContextRecallInput
) -> RagasScore:
    """
    Evaluate NonLLMContextRecall metric.
    
    Args:
        input_data: NonLLMContextRecallInput containing retrieved_contexts and reference_contexts
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    sample = SingleTurnSample(
        retrieved_contexts=input_data.retrieved_contexts,
        reference_contexts=input_data.reference_contexts
    )
    
    scorer = NonLLMContextRecall()
    result = await scorer.single_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="NonLLMContextRecall",
        id=METRIC_IDS["non_llm_context_recall"]
    )

