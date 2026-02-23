"""
TopicAdherenceScore evaluation using RAGAS.
"""
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
from ragas.metrics import TopicAdherenceScore
from ragas.llms import BaseRagasLLM
from ragas.dataset_schema import MultiTurnSample

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.utils import convert_message_list_to_ragas_messages
from app.services.evaluation.ragas.validators import convert_string_to_list
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class TopicAdherenceScoreInput(BaseModel):
    """Input type for TopicAdherenceScore evaluation."""
    user_input: List[Union[Dict[str, Any], Any]] = Field(..., description="List of message dictionaries or objects (HumanMessage, AIMessage, ToolMessage) representing multi-turn conversation")
    reference_topics: List[str] = Field(..., description="List of reference topic strings")
    
    @field_validator('reference_topics', mode='before')
    @classmethod
    def _convert_reference_topics_to_list(cls, v):
        """Convert reference_topics to list of strings."""
        return convert_string_to_list(v)
    
    @field_validator('user_input', mode='before')
    @classmethod
    def preserve_user_input(cls, v: Any) -> Any:
        """Preserve user_input as-is (dicts or objects) - conversion happens later."""
        # Just return as-is, we'll convert when creating MultiTurnSample
        return v


async def evaluate_topic_adherence_score(
    input_data: TopicAdherenceScoreInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate TopicAdherenceScore metric.
    
    Args:
        input_data: TopicAdherenceScoreInput containing user_input and reference_topics
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Convert message dictionaries to RAGAS message objects (handles dicts automatically)
    # This allows users to pass dictionaries directly without manual conversion
    messages = convert_message_list_to_ragas_messages(input_data.user_input)
    
    # Create MultiTurnSample with converted message objects
    sample = MultiTurnSample(
        user_input=messages,
        reference_topics=input_data.reference_topics
    )
    
    # Use metric directly
    scorer = TopicAdherenceScore(llm=llm) if llm else TopicAdherenceScore()
    result = await scorer.multi_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="TopicAdherenceScore",
        id=METRIC_IDS["topic_adherence_score"]
    )

