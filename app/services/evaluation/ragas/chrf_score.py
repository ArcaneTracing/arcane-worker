"""
ChrfScore evaluation using RAGAS.
"""
from pydantic import BaseModel, Field
from ragas.metrics.collections import CHRFScore

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ChrfScoreInput(BaseModel):
    """Input type for ChrfScore evaluation."""
    response: str = Field(..., description="Response string")
    reference: str = Field(..., description="Reference string")


async def evaluate_chrf_score(
    input_data: ChrfScoreInput
) -> RagasScore:
    """
    Evaluate ChrfScore metric.
    
    Uses the modern RAGAS API: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#chrf-score
    
    Args:
        input_data: ChrfScoreInput containing response and reference
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Create metric (no LLM/embeddings needed)
    scorer = CHRFScore()

    # Evaluate using modern API - accepts strings directly
    result = await scorer.ascore(
        response=input_data.response,
        reference=input_data.reference
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ChrfScore",
        id=METRIC_IDS["chrf_score"]
    )

