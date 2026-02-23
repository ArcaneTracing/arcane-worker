# Arcane

![Arcane Hero](https://arcanetracing.com/img/landing_hero_illustration.png)

**OpenTelemetry-Native Observability for AI Systems.**

[**arcanetracing.com**](https://arcanetracing.com) · [**Documentation**](https://arcanetracing.com/docs/intro) · [**Get Started Free**](https://arcanetracing.com/docs/intro) · [**Contact**](mailto:contact@arcanetracing.com)

[![PyPI arcane-sdk](https://img.shields.io/pypi/v/arcane-sdk?label=pypi%20arcane-sdk)](https://pypi.org/project/arcane-sdk/) [![npm arcane-sdk](https://img.shields.io/npm/v/arcane-sdk?label=npm%20arcane-sdk)](https://www.npmjs.com/package/arcane-sdk) [![Docker Pulls](https://img.shields.io/docker/pulls/arcanetracing/arcane?label=docker%20pulls)](https://hub.docker.com/u/arcanetracing)

---

# Arcane Worker

Worker service for the [Arcane](https://www.arcanetracing.com) platform. Consumes **experiment** and **evaluation** jobs from a message broker (RabbitMQ or Kafka), executes LLM prompts, runs RAG/agent evaluations, and publishes results back to the broker.

## Overview

This service is a FastStream + FastAPI application that:

- **Experiment jobs**: Fetches prompt versions and model configs from the Arcane API, executes LLM requests (chat completions) across supported providers, and publishes experiment results.
- **Evaluation jobs**: Runs RAGAS metrics or custom LLM-based evaluations on experiment outputs, computes scores, and publishes evaluation results.

Supported LLM providers: **OpenAI**, **Anthropic (Claude)**, **Azure OpenAI**, **AWS Bedrock**, **Google Vertex AI**, **Google AI Studio**.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- RabbitMQ *or* Kafka (depending on `MESSAGE_BROKER`)

## Installation

### Install uv (recommended)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
# Create .venv and install dependencies (generates uv.lock)
uv sync

# Include dev dependencies (pytest, coverage, etc.)
uv sync --dev

# Update dependencies
uv lock --upgrade
```

## Configuration

Copy `.env-example` to `.env` and configure:

```bash
cp .env-example .env
```

### Key settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MESSAGE_BROKER` | `rabbitmq` or `kafka` | `rabbitmq` |
| `API_BASE_URL` | Arcane API base URL | `http://localhost:8085` |
| `INTERNAL_API_KEY` | API authentication token | — |
| `EXPERIMENT_JOBS_TOPIC` | Queue/topic for experiment jobs | `experiment-queue` |
| `EVALUATION_JOBS_TOPIC` | Queue/topic for evaluation jobs | `evaluation-queue` |

### RabbitMQ

```bash
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VHOST=/
# Or use a full URL:
# RABBITMQ_URL=amqp://user:password@host:5672/vhost
```

### Kafka

```bash
KAFKA_BROKERS=localhost:9092
KAFKA_CLIENT_ID=arcane-worker
KAFKA_GROUP_ID=arcane-worker
KAFKA_SSL_ENABLED=false
KAFKA_SASL_ENABLED=false
```

### LLM provider keys

Set as needed: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_*`, `GOOGLE_*`, or AWS credentials for Bedrock.

## Running

```bash
# Run with uv
uv run python -m app.main

# Or with virtualenv activated
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m app.main
```

By default, the HTTP API listens on `http://0.0.0.0:8000`.

### Docker

```bash
docker build -t arcane-worker .
docker run -p 8000:8000 --env-file .env arcane-worker
```

## Project structure

```
app/
├── api/                    # FastAPI routes and dependencies
│   ├── routes/
│   │   ├── chat.py         # /api/v1/run - direct LLM execution
│   │   └── health.py       # /, /health
│   └── dependencies.py
├── clients/                # External API clients
│   ├── model_config_client.py
│   └── prompt_version_client.py
├── core/                   # Brokers, security, validation, registry
│   ├── broker_factory.py   # Kafka vs RabbitMQ selection
│   ├── error_handling.py
│   ├── security.py
│   └── ...
├── domain/                 # Job processors
│   ├── experiment/         # Experiment job → LLM run → result
│   └── evaluation/        # Evaluation job → RAGAS/LLM eval → score
├── models/                 # Pydantic schemas
│   └── schemas.py
├── services/
│   ├── llm/                # LLM provider implementations
│   │   ├── factory.py
│   │   ├── openai_service.py
│   │   ├── anthropic_service.py
│   │   └── ...
│   ├── evaluation/
│   │   ├── ragas/          # RAGAS evaluation metrics
│   │   ├── llm_builders/   # LLM clients for evaluation
│   │   └── llm_evaluation_processor.py
│   └── template.py         # Mustache / f-string templates
├── config.py
└── main.py                 # FastStream + FastAPI entry point
```

## API

### HTTP endpoints

- `GET /` — Root with API info
- `GET /health` — Health check
- `POST /api/v1/run` — Run a chat completion (for direct use or testing)

### API docs

When running locally:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Message broker

The worker consumes from two job queues and publishes to result queues plus dead-letter queues:

| Purpose | RabbitMQ queue / Kafka topic |
|---------|-----------------------------|
| Experiment jobs (in) | `experiment-queue` |
| Experiment results (out) | `experiment-results-queue` |
| Experiment DLQ | `experiment-dlq` |
| Evaluation jobs (in) | `evaluation-queue` |
| Evaluation results (out) | `evaluation-queue-result` |
| Evaluation DLQ | `evaluation-dlq` |

Queue/topic names are configurable via environment variables.

## Features

- **Multi-provider LLM support**: OpenAI, Anthropic, Azure, Bedrock, Google Vertex AI, Google AI Studio
- **RAGAS evaluation**: 23+ metrics for RAG/agent systems
- **Template rendering**: Mustache and f-string formats
- **Pluggable broker**: RabbitMQ or Kafka via `MESSAGE_BROKER`
- **Dead letter queues**: Failed jobs published to DLQ for retry/review
- **Security**: Input validation, JSON limits, format specifier sanitization
- **Caching**: Model configs, prompt versions, and LLM clients cached in memory

## Testing

```bash
# Run all tests with coverage
uv run pytest

# Coverage report (HTML)
uv run pytest --cov=app --cov-report=html

# Specific test file
uv run pytest tests/unit/domain/test_experiment_processor.py -v
```

Current: **263 tests**, **~68% coverage**.

## Code quality

SonarQube configuration is in `sonar-project.properties`:

```bash
# Run SonarQube analysis (requires SonarQube server)
sonar-scanner
```

## 💭 Support

- **Documentation** — [arcanetracing.com/docs](https://arcanetracing.com/docs/intro)
- **Contact** — [contact@arcanetracing.com](mailto:contact@arcanetracing.com)
- **GitHub** — [github.com/ArcaneTracing](https://github.com/ArcaneTracing)

## Built on Open Standards. Ready for Production.

Get started for free or schedule a demo to see how Arcane can transform your GenAI observability.

[**Start Free Now**](https://arcanetracing.com/docs/intro) · [**Star on GitHub**](https://github.com/ArcaneTracing)

## License

MIT © 2023–2025 ArcaneTracing
