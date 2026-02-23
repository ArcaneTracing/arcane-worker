"""
Registration of all RAGAS evaluation metrics.
This module registers all metrics with the global registry.
"""
from app.services.evaluation.ragas.base_metric import FunctionBasedMetric
from app.services.evaluation.ragas.metric_registry import register_metric
from app.services.evaluation.ragas.metric_ids import METRIC_IDS

# Import all evaluation functions and input classes
from app.services.evaluation.ragas.context_precision import (
    evaluate_context_precision,
    ContextPrecisionInput,
)
from app.services.evaluation.ragas.context_utilisation import (
    evaluate_context_utilisation,
    ContextUtilisationInput,
)
from app.services.evaluation.ragas.llm_context_recall import (
    evaluate_llm_context_recall,
    LLMContextRecallInput,
)
from app.services.evaluation.ragas.non_llm_context_recall import (
    evaluate_non_llm_context_recall,
    NonLLMContextRecallInput,
)
from app.services.evaluation.ragas.id_based_context_recall import (
    evaluate_id_based_context_recall,
    IDBasedContextRecallInput,
)
from app.services.evaluation.ragas.context_entity_recall import (
    evaluate_context_entity_recall,
    ContextEntityRecallInput,
)
from app.services.evaluation.ragas.noise_sensitivity import (
    evaluate_noise_sensitivity,
    NoiseSensitivityInput,
)
from app.services.evaluation.ragas.answer_relevancy import (
    evaluate_answer_relevancy,
    AnswerRelevancyInput,
)
from app.services.evaluation.ragas.faithfulness import (
    evaluate_faithfulness,
    FaithfulnessInput,
)
from app.services.evaluation.ragas.nvidia_answer_accuracy import (
    evaluate_nvidia_answer_accuracy,
    NvidiaAnswerAccuracyInput,
)
from app.services.evaluation.ragas.context_relevance import (
    evaluate_context_relevance,
    ContextRelevanceInput,
)
from app.services.evaluation.ragas.response_groundness import (
    evaluate_response_groundness,
    ResponseGroundnessInput,
)
from app.services.evaluation.ragas.topic_adherence_score import (
    evaluate_topic_adherence_score,
    TopicAdherenceScoreInput,
)
from app.services.evaluation.ragas.tool_call_accuracy import (
    evaluate_tool_call_accuracy,
    ToolCallAccuracyInput,
)
from app.services.evaluation.ragas.tool_call_f1 import (
    evaluate_tool_call_f1,
    ToolCallF1Input,
)
from app.services.evaluation.ragas.agent_goal_accuracy_with_reference import (
    evaluate_agent_goal_accuracy_with_reference,
    AgentGoalAccuracyWithReferenceInput,
)
from app.services.evaluation.ragas.factual_correctness import (
    evaluate_factual_correctness,
    FactualCorrectnessInput,
)
from app.services.evaluation.ragas.non_llm_string_similarity import (
    evaluate_non_llm_string_similarity,
    NonLLMStringSimilarityInput,
)
from app.services.evaluation.ragas.bleu_score import (
    evaluate_bleu_score,
    BleuScoreInput,
)
from app.services.evaluation.ragas.chrf_score import (
    evaluate_chrf_score,
    ChrfScoreInput,
)
from app.services.evaluation.ragas.rouge_score import (
    evaluate_rouge_score,
    RougeScoreInput,
)
from app.services.evaluation.ragas.string_presence import (
    evaluate_string_presence,
    StringPresenceInput,
)
from app.services.evaluation.ragas.exact_match import (
    evaluate_exact_match,
    ExactMatchInput,
)


def register_all_metrics():
    """
    Register all evaluation metrics with the global registry.
    
    This function should be called once during application startup.
    """
    # Metrics that require LLM
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["context_precision"],
        metric_name="ContextPrecision",
        input_class=ContextPrecisionInput,
        evaluation_function=evaluate_context_precision,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["context_utilisation"],
        metric_name="ContextUtilisation",
        input_class=ContextUtilisationInput,
        evaluation_function=evaluate_context_utilisation,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["llm_context_recall"],
        metric_name="LLMContextRecall",
        input_class=LLMContextRecallInput,
        evaluation_function=evaluate_llm_context_recall,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["context_entity_recall"],
        metric_name="ContextEntityRecall",
        input_class=ContextEntityRecallInput,
        evaluation_function=evaluate_context_entity_recall,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["noise_sensitivity"],
        metric_name="NoiseSensitivity",
        input_class=NoiseSensitivityInput,
        evaluation_function=evaluate_noise_sensitivity,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["answer_relevancy"],
        metric_name="AnswerRelevancy",
        input_class=AnswerRelevancyInput,
        evaluation_function=evaluate_answer_relevancy,
        requires_llm=True,
        requires_embeddings=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["faithfulness"],
        metric_name="Faithfulness",
        input_class=FaithfulnessInput,
        evaluation_function=evaluate_faithfulness,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["nvidia_answer_accuracy"],
        metric_name="NvidiaAnswerAccuracy",
        input_class=NvidiaAnswerAccuracyInput,
        evaluation_function=evaluate_nvidia_answer_accuracy,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["context_relevance"],
        metric_name="ContextRelevance",
        input_class=ContextRelevanceInput,
        evaluation_function=evaluate_context_relevance,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["response_groundness"],
        metric_name="ResponseGroundness",
        input_class=ResponseGroundnessInput,
        evaluation_function=evaluate_response_groundness,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["topic_adherence_score"],
        metric_name="TopicAdherenceScore",
        input_class=TopicAdherenceScoreInput,
        evaluation_function=evaluate_topic_adherence_score,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["agent_goal_accuracy_with_reference"],
        metric_name="AgentGoalAccuracyWithReference",
        input_class=AgentGoalAccuracyWithReferenceInput,
        evaluation_function=evaluate_agent_goal_accuracy_with_reference,
        requires_llm=True
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["factual_correctness"],
        metric_name="FactualCorrectness",
        input_class=FactualCorrectnessInput,
        evaluation_function=evaluate_factual_correctness,
        requires_llm=True
    ))
    
    # Metrics that don't require LLM or embeddings
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["non_llm_context_recall"],
        metric_name="NonLLMContextRecall",
        input_class=NonLLMContextRecallInput,
        evaluation_function=evaluate_non_llm_context_recall,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["id_based_context_recall"],
        metric_name="IDBasedContextRecall",
        input_class=IDBasedContextRecallInput,
        evaluation_function=evaluate_id_based_context_recall,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["tool_call_accuracy"],
        metric_name="ToolCallAccuracy",
        input_class=ToolCallAccuracyInput,
        evaluation_function=evaluate_tool_call_accuracy,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["tool_call_f1"],
        metric_name="ToolCallF1",
        input_class=ToolCallF1Input,
        evaluation_function=evaluate_tool_call_f1,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["non_llm_string_similarity"],
        metric_name="NonLLMStringSimilarity",
        input_class=NonLLMStringSimilarityInput,
        evaluation_function=evaluate_non_llm_string_similarity,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["bleu_score"],
        metric_name="BleuScore",
        input_class=BleuScoreInput,
        evaluation_function=evaluate_bleu_score,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["chrf_score"],
        metric_name="ChrfScore",
        input_class=ChrfScoreInput,
        evaluation_function=evaluate_chrf_score,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["rouge_score"],
        metric_name="RougeScore",
        input_class=RougeScoreInput,
        evaluation_function=evaluate_rouge_score,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["string_presence"],
        metric_name="StringPresence",
        input_class=StringPresenceInput,
        evaluation_function=evaluate_string_presence,
        requires_llm=False
    ))
    
    register_metric(FunctionBasedMetric(
        metric_id=METRIC_IDS["exact_match"],
        metric_name="ExactMatch",
        input_class=ExactMatchInput,
        evaluation_function=evaluate_exact_match,
        requires_llm=False
    ))


# Auto-register all metrics when this module is imported
register_all_metrics()

