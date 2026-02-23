"""
AnswerRelevancy evaluation using RAGAS.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from ragas.metrics.collections import AnswerRelevancy
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.validators import convert_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class AnswerRelevancyInput(BaseModel):
    """Input type for AnswerRelevancy evaluation."""
    user_input: str = Field(..., description="User input string")
    response: str = Field(..., description="Response string")
    
    @field_validator('user_input', 'response', mode='before')
    @classmethod
    def _convert_to_string(cls, v):
        """Convert value to string (handles int, float, None, etc.)."""
        return convert_to_string(v)


async def evaluate_answer_relevancy(
    input_data: AnswerRelevancyInput,
    llm: Optional[BaseRagasLLM] = None,
    embeddings: Optional[BaseRagasEmbedding] = None
) -> RagasScore:
    """
    Evaluate AnswerRelevancy metric.
    
    Args:
        input_data: AnswerRelevancyInput containing user_input and response
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        embeddings: Optional embeddings instance (required for AnswerRelevancy metric)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for AnswerRelevancy metric")
    if embeddings is None:
        raise ValueError("Embeddings is required for AnswerRelevancy metric")
    
    # Create metric instance
    scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    
    # Evaluate single input
    result = await scorer.ascore(
        user_input=input_data.user_input,
        response=input_data.response
    )
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="AnswerRelevancy",
        id=METRIC_IDS["answer_relevancy"]
    )

