"""
StringPresence evaluation using RAGAS.
"""
from typing import List
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import StringPresence

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_list_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class StringPresenceInput(BaseModel):
    """Input type for StringPresence evaluation."""
    reference: str | List[str] = Field(..., description="Reference string or list of reference strings")
    response: str | List[str] = Field(..., description="Response string or list of response strings")
    
    @field_validator('reference', 'response', mode='before')
    @classmethod
    def _convert_list_to_string(cls, v):
        """Convert list to string (take first element) or preserve string."""
        return convert_list_to_string(v)


async def evaluate_string_presence(
    input_data: StringPresenceInput
) -> RagasScore:
    """
    Evaluate StringPresence metric.
    
    Uses the modern RAGAS API: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#string-presence
    
    Args:
        input_data: StringPresenceInput containing reference and response
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Create metric (no LLM/embeddings needed)
    scorer = StringPresence()
    
    # Evaluate using modern API - accepts strings directly
    result = await scorer.ascore(
        reference=input_data.reference,
        response=input_data.response
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="StringPresence",
        id=METRIC_IDS["string_presence"]
    )

