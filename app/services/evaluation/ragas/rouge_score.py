"""
RougeScore evaluation using RAGAS.
"""
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import RougeScore
from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class RougeScoreInput(BaseModel):
    """Input type for RougeScore evaluation."""
    reference: str = Field(..., description="Reference string")
    response: str = Field(..., description="Response string")
    
    @field_validator('reference', 'response', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)


async def evaluate_rouge_score(
    input_data: RougeScoreInput
) -> RagasScore:
    """
    Evaluate RougeScore metric.
    
    Args:
        input_data: RougeScoreInput containing reference and response
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Create metric (no LLM/embeddings needed)
    scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")
    
    # Evaluate using modern API - accepts strings directly
    result = await scorer.ascore(
        reference=input_data.reference,
        response=input_data.response
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="RougeScore",
        id=METRIC_IDS["rouge_score"]
    )

