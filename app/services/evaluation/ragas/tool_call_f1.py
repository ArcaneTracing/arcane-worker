"""
ToolCallF1 evaluation using RAGAS.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ragas.metrics import ToolCallF1
from ragas.dataset_schema import MultiTurnSample

from app.services.evaluation.ragas.models.ragas_score import RagasScore
from app.services.evaluation.ragas.utils import (
    convert_message_list_to_ragas_messages,
    convert_tool_call_list_to_ragas_tool_calls
)
from app.services.evaluation.ragas.helpers import extract_score
from app.services.evaluation.ragas.metric_ids import METRIC_IDS


class ToolCallF1Input(BaseModel):
    """Input type for ToolCallF1 evaluation."""
    user_input: List[Dict[str, Any]] = Field(..., description="List of message dictionaries (HumanMessage, AIMessage, ToolMessage) representing multi-turn conversation")
    reference_tool_calls: List[Dict[str, Any]] = Field(..., description="List of reference tool call dictionaries")


async def evaluate_tool_call_f1(
    input_data: ToolCallF1Input
) -> RagasScore:
    """
    Evaluate ToolCallF1 metric.
    
    Args:
        input_data: ToolCallF1Input containing user_input and reference_tool_calls
        
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
    scorer = ToolCallF1()
    result = await scorer.multi_turn_ascore(sample)
    
    score = extract_score(result)
    
    return RagasScore(
        score=score,
        metric="ToolCallF1",
        id=METRIC_IDS["tool_call_f1"]
    )

