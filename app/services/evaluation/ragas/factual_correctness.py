"""
FactualCorrectness evaluation using RAGAS.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import FactualCorrectness
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class FactualCorrectnessInput(BaseModel):
    """Input type for FactualCorrectness evaluation."""
    response: str = Field(..., description="Response string")
    reference: str = Field(..., description="Reference string")
    
    @field_validator('response', 'reference', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)


async def evaluate_factual_correctness(
    input_data: FactualCorrectnessInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate FactualCorrectness metric.
    
    Args:
        input_data: FactualCorrectnessInput containing response and reference
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for FactualCorrectness metric")
    
    # Create metric instance
    scorer = FactualCorrectness(llm=llm)
    
    # Evaluate single input
    result = await scorer.ascore(
        response=input_data.response,
        reference=input_data.reference
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="FactualCorrectness",
        id=METRIC_IDS["factual_correctness"]
    )

