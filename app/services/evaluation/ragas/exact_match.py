"""
ExactMatch evaluation using RAGAS.
"""
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import ExactMatch

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ExactMatchInput(BaseModel):
    """Input type for ExactMatch evaluation."""
    reference: str = Field(..., description="Reference string")
    response: str = Field(..., description="Response string")
    
    @field_validator('reference', 'response', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)


async def evaluate_exact_match(
    input_data: ExactMatchInput
) -> RagasScore:
    """
    Evaluate ExactMatch metric.
    
    Args:
        input_data: ExactMatchInput containing reference and response
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Create metric (no LLM/embeddings needed)
    scorer = ExactMatch()
    
    # Evaluate using modern API - accepts strings directly
    result = await scorer.ascore(
        reference=input_data.reference,
        response=input_data.response
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ExactMatch",
        id=METRIC_IDS["exact_match"]
    )

