"""
RAGAS evaluation metrics.
"""
# Import metrics registration to ensure all metrics are registered
from app.services.evaluation.ragas import metrics_registration  # noqa: F401

from app.services.evaluation.ragas.context_precision import evaluate_context_precision
from app.services.evaluation.ragas.context_utilisation import evaluate_context_utilisation
from app.services.evaluation.ragas.llm_context_recall import evaluate_llm_context_recall
from app.services.evaluation.ragas.non_llm_context_recall import evaluate_non_llm_context_recall
from app.services.evaluation.ragas.id_based_context_recall import evaluate_id_based_context_recall
from app.services.evaluation.ragas.context_entity_recall import evaluate_context_entity_recall
from app.services.evaluation.ragas.noise_sensitivity import evaluate_noise_sensitivity
from app.services.evaluation.ragas.answer_relevancy import evaluate_answer_relevancy
from app.services.evaluation.ragas.faithfulness import evaluate_faithfulness
from app.services.evaluation.ragas.nvidia_answer_accuracy import evaluate_nvidia_answer_accuracy
from app.services.evaluation.ragas.context_relevance import evaluate_context_relevance
from app.services.evaluation.ragas.response_groundness import evaluate_response_groundness
from app.services.evaluation.ragas.topic_adherence_score import evaluate_topic_adherence_score
from app.services.evaluation.ragas.tool_call_accuracy import evaluate_tool_call_accuracy
from app.services.evaluation.ragas.tool_call_f1 import evaluate_tool_call_f1
from app.services.evaluation.ragas.agent_goal_accuracy_with_reference import evaluate_agent_goal_accuracy_with_reference
from app.services.evaluation.ragas.factual_correctness import evaluate_factual_correctness
from app.services.evaluation.ragas.non_llm_string_similarity import evaluate_non_llm_string_similarity
from app.services.evaluation.ragas.bleu_score import evaluate_bleu_score
from app.services.evaluation.ragas.chrf_score import evaluate_chrf_score
from app.services.evaluation.ragas.rouge_score import evaluate_rouge_score
from app.services.evaluation.ragas.string_presence import evaluate_string_presence
from app.services.evaluation.ragas.exact_match import evaluate_exact_match

__all__ = [
    "evaluate_context_precision",
    "evaluate_context_utilisation",
    "evaluate_llm_context_recall",
    "evaluate_non_llm_context_recall",
    "evaluate_id_based_context_recall",
    "evaluate_context_entity_recall",
    "evaluate_noise_sensitivity",
    "evaluate_answer_relevancy",
    "evaluate_faithfulness",
    "evaluate_nvidia_answer_accuracy",
    "evaluate_context_relevance",
    "evaluate_response_groundness",
    "evaluate_topic_adherence_score",
    "evaluate_tool_call_accuracy",
    "evaluate_tool_call_f1",
    "evaluate_agent_goal_accuracy_with_reference",
    "evaluate_factual_correctness",
    "evaluate_non_llm_string_similarity",
    "evaluate_bleu_score",
    "evaluate_chrf_score",
    "evaluate_rouge_score",
    "evaluate_string_presence",
    "evaluate_exact_match",
    "EVALUATION_FUNCTION_TO_ID",
]

from app.services.evaluation.ragas.metric_ids import METRIC_IDS

# Mapping of evaluation function names to their metric IDs
# Function names map to metric names, which are then looked up in METRIC_IDS
_FUNCTION_TO_METRIC_NAME = {
    "evaluate_context_precision": "context_precision",
    "evaluate_context_utilisation": "context_utilisation",
    "evaluate_llm_context_recall": "llm_context_recall",
    "evaluate_non_llm_context_recall": "non_llm_context_recall",
    "evaluate_id_based_context_recall": "id_based_context_recall",
    "evaluate_context_entity_recall": "context_entity_recall",
    "evaluate_noise_sensitivity": "noise_sensitivity",
    "evaluate_answer_relevancy": "answer_relevancy",
    "evaluate_faithfulness": "faithfulness",
    "evaluate_nvidia_answer_accuracy": "nvidia_answer_accuracy",
    "evaluate_context_relevance": "context_relevance",
    "evaluate_response_groundness": "response_groundness",
    "evaluate_topic_adherence_score": "topic_adherence_score",
    "evaluate_tool_call_accuracy": "tool_call_accuracy",
    "evaluate_tool_call_f1": "tool_call_f1",
    "evaluate_agent_goal_accuracy_with_reference": "agent_goal_accuracy_with_reference",
    "evaluate_factual_correctness": "factual_correctness",
    "evaluate_non_llm_string_similarity": "non_llm_string_similarity",
    "evaluate_bleu_score": "bleu_score",
    "evaluate_chrf_score": "chrf_score",
    "evaluate_rouge_score": "rouge_score",
    "evaluate_string_presence": "string_presence",
    "evaluate_exact_match": "exact_match",
}

EVALUATION_FUNCTION_TO_ID = {
    func_name: METRIC_IDS[metric_name]
    for func_name, metric_name in _FUNCTION_TO_METRIC_NAME.items()
}

