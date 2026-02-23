"""
Application configuration settings.

Uses Pydantic Settings to load configuration from environment variables.
Supports loading from .env file for local development.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Literal, Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Supports both .env file (for local development) and environment variables
    (for production deployments).
    
    All settings can be overridden via environment variables.
    """
    CORS_ORIGINS: List[str] = ["*"]

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR",
    )
    
    MESSAGE_BROKER: Literal["rabbitmq", "kafka"] = Field(
        default="rabbitmq",
        description="Message broker: rabbitmq or kafka",
    )

    RABBITMQ_URL: Optional[str] = Field(None, description="RabbitMQ connection URL (amqp://user:password@host:port/vhost)")
    RABBITMQ_HOST: str = Field(default="localhost", description="RabbitMQ host")
    RABBITMQ_PORT: int = Field(default=5672, description="RabbitMQ port")
    RABBITMQ_USERNAME: str = Field(default="guest", description="RabbitMQ username")
    RABBITMQ_PASSWORD: str = Field(default="guest", description="RabbitMQ password")
    RABBITMQ_VHOST: str = Field(default="/", description="RabbitMQ virtual host")

    # Kafka (when MESSAGE_BROKER=kafka)
    KAFKA_BROKERS: str = Field(
        default="localhost:9092",
        description="Comma-separated Kafka broker list",
    )
    KAFKA_CLIENT_ID: str = Field(default="arcane-worker", description="Kafka client ID")
    KAFKA_GROUP_ID: str = Field(
        default="arcane-worker",
        description="Kafka consumer group ID",
    )
    KAFKA_SSL_ENABLED: bool = Field(default=False, description="Use SSL/TLS for Kafka")
    KAFKA_SASL_ENABLED: bool = Field(default=False, description="Use SASL auth for Kafka")
    KAFKA_SASL_MECHANISM: str = Field(
        default="plain",
        description="SASL mechanism: plain, scram-sha-256, scram-sha-512",
    )
    KAFKA_SASL_USERNAME: Optional[str] = Field(None, description="Kafka SASL username")
    KAFKA_SASL_PASSWORD: Optional[str] = Field(None, description="Kafka SASL password")

    # Worker settings
    WORKER_POLL_INTERVAL_SECONDS: int = Field(default=10, description="Interval between queue polls")
    WORKER_MAX_CONCURRENT_JOBS: int = Field(default=3, description="Maximum concurrent job processing")
    WORKER_JOB_LOCK_TTL_SECONDS: int = Field(default=300, description="Job lock TTL in seconds")

    # Queue/topic names (RabbitMQ queues, Kafka topics). Must match backend config.
    EXPERIMENT_JOBS_TOPIC: str = Field(
        default="experiment-queue",
        description="Queue/topic for experiment jobs (listen on)",
    )
    EXPERIMENT_RESULTS_TOPIC: str = Field(
        default="experiment-results-queue",
        description="Queue/topic for experiment results (publish to)",
    )
    EXPERIMENT_DLQ_QUEUE: str = Field(
        default="experiment-dlq",
        description="Dead letter queue for failed experiment jobs",
    )
    EVALUATION_JOBS_TOPIC: str = Field(
        default="evaluation-queue",
        description="Queue/topic for evaluation jobs (listen on)",
    )
    EVALUATION_RESULTS_TOPIC: str = Field(
        default="evaluation-queue-result",
        description="Queue/topic for evaluation results (publish to)",
    )
    EVALUATION_DLQ_QUEUE: str = Field(
        default="evaluation-dlq",
        description="Dead letter queue for failed evaluation jobs",
    )
    
    # API settings
    API_BASE_URL: str = Field(default="http://localhost:8085", description="Base URL for API requests")
    INTERNAL_API_KEY: Optional[str] = Field(None, description="JWT token for API authentication")
    API_TIMEOUT_SECONDS: float = Field(default=30.0, description="Timeout for API requests (connect + read)")

    # In-memory cache TTL (seconds) - reduces redundant API calls and client instantiation
    CACHE_MODEL_CONFIG_TTL: int = Field(default=300, description="Model config cache TTL (seconds)")
    CACHE_PROMPT_VERSION_TTL: int = Field(default=300, description="Prompt version cache TTL (seconds)")
    CACHE_MODEL_SERVICE_TTL: int = Field(default=600, description="Built LLM client cache TTL (seconds)")
    CACHE_MODEL_SERVICE_MAX_SIZE: int = Field(default=50, description="Max cached LLM clients")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

