"""
NoiseSensitivity evaluation using RAGAS.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import NoiseSensitivity
from ragas.llms import BaseRagasLLM

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string, convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class NoiseSensitivityInput(BaseModel):
    """Input type for NoiseSensitivity evaluation."""
    user_input: str = Field(..., description="User input string")
    response: str = Field(..., description="Response string")
    reference: str = Field(..., description="Reference string")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    
    @field_validator('user_input', 'response', 'reference', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)
    
    @field_validator('retrieved_contexts', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        """Convert single string to single-item list if needed."""
        return convert_string_to_list(v)


async def evaluate_noise_sensitivity(
    input_data: NoiseSensitivityInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate NoiseSensitivity metric.
    
    Args:
        input_data: NoiseSensitivityInput containing user_input, response, reference, and retrieved_contexts
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for NoiseSensitivity metric")
    
    # Create metric instance
    scorer = NoiseSensitivity(llm=llm)
    
    # Evaluate single input
    result = await scorer.ascore(
        user_input=input_data.user_input,
        response=input_data.response,
        reference=input_data.reference,
        retrieved_contexts=input_data.retrieved_contexts
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="NoiseSensitivity",
        id=METRIC_IDS["noise_sensitivity"]
    )

