"""
IDBasedContextRecall evaluation using RAGAS.
"""
from typing import List
from pydantic import BaseModel, Field, field_validator
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextRecall

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class IDBasedContextRecallInput(BaseModel):
    """Input type for IDBasedContextRecall evaluation."""
    retrieved_context_ids: List[str] = Field(..., description="List of retrieved context IDs")
    reference_context_ids: List[str] = Field(..., description="List of reference context IDs")

    @field_validator('retrieved_context_ids', 'reference_context_ids', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)
async def evaluate_id_based_context_recall(
    input_data: IDBasedContextRecallInput
) -> RagasScore:
    """
    Evaluate IDBasedContextRecall metric.
    
    Args:
        input_data: IDBasedContextRecallInput containing retrieved_context_ids and reference_context_ids
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    sample = SingleTurnSample(
        retrieved_context_ids=input_data.retrieved_context_ids,
        reference_context_ids=input_data.reference_context_ids
    )
    
    scorer = IDBasedContextRecall()
    result = await scorer.single_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="IDBasedContextRecall",
        id=METRIC_IDS["id_based_context_recall"]
    )

