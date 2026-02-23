"""
NonLLMStringSimilarity evaluation using RAGAS.
"""
from typing import List
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import NonLLMStringSimilarity

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_list_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class NonLLMStringSimilarityInput(BaseModel):
    """Input type for NonLLMStringSimilarity evaluation."""
    reference: str | List[str] = Field(..., description="Reference string or list of reference strings")
    response: str | List[str] = Field(..., description="Response string or list of response strings")
    
    @field_validator('reference', 'response', mode='before')
    @classmethod
    def _convert_list_to_string(cls, v):
        """Convert list to string (take first element) or preserve string."""
        return convert_list_to_string(v)


async def evaluate_non_llm_string_similarity(
    input_data: NonLLMStringSimilarityInput
) -> RagasScore:
    """
    Evaluate NonLLMStringSimilarity metric.
    
    Uses the modern RAGAS API: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/traditional/#non-llm-string-similarity
    
    Args:
        input_data: NonLLMStringSimilarityInput containing reference and response
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Create metric (no LLM/embeddings needed)
    scorer = NonLLMStringSimilarity()
    
    # Evaluate using modern API - accepts strings directly
    result = await scorer.ascore(
        reference=input_data.reference,
        response=input_data.response
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="NonLLMStringSimilarity",
        id=METRIC_IDS["non_llm_string_similarity"]
    )

