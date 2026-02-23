"""
Metric IDs (UUIDs) for RAGAS evaluation functions.
Centralized location for all metric identifiers.
"""
from typing import Dict, Literal

# Type alias for metric ID
MetricID = str

# Mapping of metric names to their UUIDs
METRIC_IDS: Dict[str, MetricID] = {
    "context_precision": "16070b12-c49f-4b3b-904f-533e4022d0f4",
    "context_utilisation": "04b24ff4-a6f2-4d41-bde7-76bf7bd668b1",
    "llm_context_recall": "c90f0364-481b-4bd5-a162-50897300d5e1",
    "non_llm_context_recall": "4c11c803-6039-49c8-8c34-24154c92db49",
    "id_based_context_recall": "a41e87f7-2061-4d8e-83de-f49d219d450e",
    "context_entity_recall": "848d2504-19ae-4f9c-954e-366926206034",
    "noise_sensitivity": "d57720d3-4d82-40c8-9e17-a4768e1f5126",
    "answer_relevancy": "54056e81-7f85-46a9-98a5-81b75b7a0d4c",
    "faithfulness": "ea8c28c7-18c1-40c7-bdf6-b834eddb5be1",
    "nvidia_answer_accuracy": "99795951-de16-44cb-8c1d-71d100ed36d2",
    "context_relevance": "1e819871-81f4-4248-a681-c653e1910939",
    "response_groundness": "7cadef69-f9b8-4dc7-b9eb-7245b1e3a3dc",
    "topic_adherence_score": "5b0b6ec0-4a87-407f-a032-c2d5f9511569",
    "tool_call_accuracy": "a52fe55c-5a19-476d-be53-6844891ee435",
    "tool_call_f1": "5d9ea1e9-9f07-43d3-b22b-0431383bdd9d",
    "agent_goal_accuracy_with_reference": "0b0dcd87-1c11-41e9-ad93-39911014c120",
    "factual_correctness": "0d5a785c-748b-465f-b96f-6f9c7604f645",
    "non_llm_string_similarity": "6eb8a58c-4419-4500-bdec-7b9b2f3fa5a8",
    "bleu_score": "34704532-c2c8-403e-8114-09755a281969",
    "chrf_score": "edea8a30-999c-4b0e-9779-dc680d83a9c0",
    "rouge_score": "b75661d8-0d69-41d2-abdb-f99800b98391",
    "string_presence": "a0fed9fb-075f-4c9d-9be9-093879ec04ad",
    "exact_match": "ceeaa1d8-58d0-4bde-9bb6-b9380163b706",
}


def get_metric_id(metric_name: str) -> MetricID:
    """
    Get metric ID by metric name.
    
    Args:
        metric_name: Name of the metric (e.g., "llm_context_recall")
        
    Returns:
        Metric ID (UUID string)
        
    Raises:
        KeyError: If metric name is not found
    """
    return METRIC_IDS[metric_name]

