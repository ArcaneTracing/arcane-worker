"""
AgentGoalAccuracyWithReference evaluation using RAGAS.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from ragas.metrics import AgentGoalAccuracyWithReference
from ragas.llms import BaseRagasLLM
from ragas.dataset_schema import MultiTurnSample

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.utils import convert_message_list_to_ragas_messages
from app.services.evaluation.ragas.validators import convert_list_to_string
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class AgentGoalAccuracyWithReferenceInput(BaseModel):
    """Input type for AgentGoalAccuracyWithReference evaluation."""
    user_input: List[Dict[str, Any]] = Field(..., description="List of message dictionaries (HumanMessage, AIMessage, ToolMessage) representing multi-turn conversation")
    reference: str = Field(..., description="Reference string describing the expected goal/outcome")
    
    @field_validator('reference', mode='before')
    @classmethod
    def _convert_reference_to_string(cls, v):
        """Convert reference to string if it's a list (take first element)."""
        return convert_list_to_string(v)


async def evaluate_agent_goal_accuracy_with_reference(
    input_data: AgentGoalAccuracyWithReferenceInput,
    llm: Optional[BaseRagasLLM] = None
) -> RagasScore:
    """
    Evaluate AgentGoalAccuracyWithReference metric.
    
    Args:
        input_data: AgentGoalAccuracyWithReferenceInput containing user_input and reference
        llm: Optional LLM instance for evaluation (required for LLM-based metrics)
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    if llm is None:
        raise ValueError("LLM is required for AgentGoalAccuracyWithReference metric")
    
    # Convert message dictionaries to RAGAS message objects (handles dicts automatically)
    messages = convert_message_list_to_ragas_messages(input_data.user_input)
    
    # Create MultiTurnSample as required by RAGAS API
    sample = MultiTurnSample(
        user_input=messages,
        reference=input_data.reference
    )
    
    # Use metric directly
    metric = AgentGoalAccuracyWithReference(llm=llm)
    result = await metric.multi_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="AgentGoalAccuracyWithReference",
        id=METRIC_IDS["agent_goal_accuracy_with_reference"]
    )

