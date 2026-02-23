"""
ToolCallAccuracy evaluation using RAGAS.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ragas.metrics import ToolCallAccuracy
from ragas.dataset_schema import MultiTurnSample

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.utils import (
    convert_message_list_to_ragas_messages,
    convert_tool_call_list_to_ragas_tool_calls
)
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ToolCallAccuracyInput(BaseModel):
    """Input type for ToolCallAccuracy evaluation."""
    user_input: List[Dict[str, Any]] = Field(..., description="List of message dictionaries (HumanMessage, AIMessage, ToolMessage) representing multi-turn conversation")
    reference_tool_calls: List[Dict[str, Any]] = Field(..., description="List of reference tool call dictionaries")


async def evaluate_tool_call_accuracy(
    input_data: ToolCallAccuracyInput
) -> RagasScore:
    """
    Evaluate ToolCallAccuracy metric.
    
    Args:
        input_data: ToolCallAccuracyInput containing user_input and reference_tool_calls
        
    Returns:
        RagasScore containing evaluation results with metric ID and score
    """
    # Convert message dictionaries to RAGAS message objects (handles dicts automatically)
    messages = convert_message_list_to_ragas_messages(input_data.user_input)
    
    # Convert reference_tool_calls to ToolCall objects (handles dicts automatically)
    reference_tool_calls = convert_tool_call_list_to_ragas_tool_calls(input_data.reference_tool_calls)
    
    # Create MultiTurnSample - accepts dictionaries and converts them automatically
    sample = MultiTurnSample(
        user_input=messages,
        reference_tool_calls=reference_tool_calls
    )
    
    # Use metric directly
    scorer = ToolCallAccuracy()
    result = await scorer.multi_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ToolCallAccuracy",
        id=METRIC_IDS["tool_call_accuracy"]
    )

