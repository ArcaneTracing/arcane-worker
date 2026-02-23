"""
Topic names for Kafka (align with NestJS topic-config.ts).
When MESSAGE_BROKER=kafka, these are used for consume and publish.
"""
EXPERIMENT_JOBS_TOPIC = "experiment-jobs"
EXPERIMENT_RESULTS_TOPIC = "experiment-results"
EXPERIMENT_DLQ_TOPIC = "experiment-dlq"
EXPERIMENT_JOBS_TOPIC = "evaluation-jobs"
EVALUATION_RESULTS_TOPIC = "evaluation-results"
EVALUATION_DLQ_TOPIC = "evaluation-dlq"
