"""
Pydantic models and schemas for the application.

Defines all data transfer objects (DTOs) used for:
- LLM service requests and responses
- Model configurations
- Prompt templates and versions
- Evaluation job data
- Experiment job data
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Literal, Union, Dict, Any, TypedDict
from datetime import datetime
from enum import Enum

# ============================================================================
# LLM Service Types - Enums
# ============================================================================

class AdapterType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    GOOGLE_VERTEX_AI = "google-vertex-ai"
    GOOGLE_AI_STUDIO = "google-ai-studio"


class TemplateType(str, Enum):
    CHAT = "CHAT"
    STR = "STR"


class TemplateFormat(str, Enum):
    MUSTACHE = "MUSTACHE"
    F_STRING = "F_STRING"
    NONE = "NONE"


# ============================================================================
# Field Description Constants
# ============================================================================

DATASET_ROW_ID_DESCRIPTION = "UUID of the dataset row"


# ============================================================================
# Model Configuration Types
# ============================================================================

class OpenAIConfig(BaseModel):
    base_url: Optional[str] = Field(None, description="Custom base URL")
    organization: Optional[str] = Field(None, description="Organization ID")


class AzureOpenAIConfig(BaseModel):
    endpoint: str = Field(..., description="Azure endpoint")
    api_version: str = Field(default="2024-02-15-preview", description="API version")
    deployment_name: Optional[str] = Field(None, description="Deployment name")


class AnthropicConfig(BaseModel):
    base_url: Optional[str] = Field(None, description="Custom base URL")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")


class BaseModelConfiguration(BaseModel):
    adapter: AdapterType = Field(..., description="Adapter type")
    model_name: str = Field(..., alias="modelName", description="Model name")
    api_key: str = Field(..., alias="apiKey", description="API key (decrypted by backend for LLM calls)")
    input_cost_per_token: Optional[float] = Field(None, alias="inputCostPerToken", description="Input cost per token")
    output_cost_per_token: Optional[float] = Field(None, alias="outputCostPerToken", description="Output cost per token")
    temperature: Optional[float] = Field(None, description="Temperature")
    max_tokens: Optional[int] = Field(None, alias="maxTokens", description="Max tokens")
    top_p: Optional[float] = Field(None, alias="topP", description="Top P")
    frequency_penalty: Optional[float] = Field(None, alias="frequencyPenalty", description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, alias="presencePenalty", description="Presence penalty")
    stop_sequences: Optional[List[str]] = Field(None, alias="stopSequences", description="Stop sequences")
    config: Optional[Dict[str, Any]] = Field(None, description="Provider-specific config")
    
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


ModelConfigurationData = BaseModelConfiguration


class ModelConfigurationWithEncryptedKey(BaseModel):
    id: str = Field(..., description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    configuration: BaseModelConfiguration = Field(..., description="Model configuration")
    created_at: str = Field(..., alias="createdAt", description="Creation timestamp (ISO 8601)")
    updated_at: str = Field(..., alias="updatedAt", description="Update timestamp (ISO 8601)")
    
    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Prompt Template Types
# ============================================================================

class TextContentPart(BaseModel):
    text: str = Field(..., description="Text content")


class ToolCallContentPart(BaseModel):
    tool_call: Dict[str, Any] = Field(..., description="Tool call information")


class ToolResultContentPart(BaseModel):
    tool_result: Dict[str, Any] = Field(..., description="Tool result")


ContentPart = Union[TextContentPart, ToolCallContentPart, ToolResultContentPart]


class PromptMessage(BaseModel):
    role: Literal["user", "assistant", "model", "ai", "tool", "system", "developer"] = Field(
        ..., description="Message role"
    )
    content: str | List[ContentPart] = Field(..., description="Message content")


class PromptStringTemplate(BaseModel):
    type: Literal["string"] = "string"
    template: str = Field(..., description="String template")


class PromptChatTemplate(BaseModel):
    type: Literal["chat"] = "chat"
    messages: List[PromptMessage] = Field(..., description="Chat messages")


PromptTemplate = Union[PromptStringTemplate, PromptChatTemplate]


# ============================================================================
# Invocation Parameters Types
# ============================================================================

class OpenAIInvocationParameters(BaseModel):
    type: Literal["openai"] = "openai"
    openai: Dict[str, Any] = Field(..., description="OpenAI parameters")


class AzureOpenAIInvocationParameters(BaseModel):
    type: Literal["azure_openai"] = "azure_openai"
    azure_openai: Dict[str, Any] = Field(..., description="Azure OpenAI parameters")


class AnthropicInvocationParameters(BaseModel):
    type: Literal["anthropic"] = "anthropic"
    anthropic: Dict[str, Any] = Field(..., description="Anthropic parameters")


class GoogleInvocationParameters(BaseModel):
    type: Literal["google"] = "google"
    google: Dict[str, Any] = Field(..., description="Google parameters")


class GoogleAIStudioInvocationParameters(BaseModel):
    type: Literal["google-ai-studio"] = "google-ai-studio"
    google_ai_studio: Dict[str, Any] = Field(
        ..., alias="google-ai-studio", description="Google AI Studio parameters"
    )

    model_config = ConfigDict(populate_by_name=True)


class GoogleVertexAIInvocationParameters(BaseModel):
    type: Literal["google-vertex-ai"] = "google-vertex-ai"
    google_vertex_ai: Dict[str, Any] = Field(
        ..., alias="google-vertex-ai", description="Google Vertex AI parameters"
    )

    model_config = ConfigDict(populate_by_name=True)


class DeepSeekInvocationParameters(BaseModel):
    type: Literal["deepseek"] = "deepseek"
    deepseek: Dict[str, Any] = Field(..., description="DeepSeek parameters")


class XAIInvocationParameters(BaseModel):
    type: Literal["xai"] = "xai"
    xai: Dict[str, Any] = Field(..., description="XAI parameters")


class OllamaInvocationParameters(BaseModel):
    type: Literal["ollama"] = "ollama"
    ollama: Dict[str, Any] = Field(..., description="Ollama parameters")


class AwsInvocationParameters(BaseModel):
    type: Literal["aws"] = "aws"
    aws: Dict[str, Any] = Field(..., description="AWS parameters")


class BedrockInvocationParameters(BaseModel):
    type: Literal["bedrock"] = "bedrock"
    bedrock: Dict[str, Any] = Field(..., description="Bedrock parameters")


InvocationParameters = Union[
    OpenAIInvocationParameters,
    AzureOpenAIInvocationParameters,
    AnthropicInvocationParameters,
    GoogleInvocationParameters,
    GoogleAIStudioInvocationParameters,
    GoogleVertexAIInvocationParameters,
    DeepSeekInvocationParameters,
    XAIInvocationParameters,
    OllamaInvocationParameters,
    AwsInvocationParameters,
    BedrockInvocationParameters,
]


# ============================================================================
# Tools Types
# ============================================================================

class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


class ToolChoiceZeroOrMore(BaseModel):
    type: Literal["zero_or_more"] = "zero_or_more"


class ToolChoiceOneOrMore(BaseModel):
    type: Literal["one_or_more"] = "one_or_more"


class ToolChoiceSpecificFunction(BaseModel):
    type: Literal["specific_function"] = "specific_function"
    function_name: str = Field(..., description="Function name")


ToolChoice = Union[
    ToolChoiceNone,
    ToolChoiceZeroOrMore,
    ToolChoiceOneOrMore,
    ToolChoiceSpecificFunction,
]


class Tools(BaseModel):
    """Tools pass-through: user provides provider-specific format; no normalization."""
    type: Literal["tools"] = "tools"
    tools: List[Dict[str, Any]] = Field(..., description="List of tools (provider-specific format, pass-through)")
    tool_choice: Optional[ToolChoice] = Field(None, description="Tool choice")
    disable_parallel_tool_calls: Optional[bool] = Field(None, description="Disable parallel tool calls")


# ============================================================================
# Response Format Types
# ============================================================================
# Pass-through: user provides provider-specific format (OpenAI, Anthropic, etc.)


# ============================================================================
# Prompt Version DTO
# ============================================================================

class PromptVersionDto(BaseModel):
    id: Optional[str] = Field(None, description="Prompt version ID")
    prompt_id: Optional[str] = Field(None, alias="promptId", description="Prompt ID")
    prompt_name: Optional[str] = Field(None, alias="promptName", description="Prompt name")
    version_name: Optional[str] = Field(None, alias="versionName", description="Version name")
    description: Optional[str] = Field(None, description="Description")
    model_configuration_id: str = Field(..., alias="modelConfigurationId", description="Model configuration ID")
    template: PromptTemplate = Field(..., description="Prompt template")
    template_type: TemplateType = Field(..., alias="templateType", description="Template type")
    template_format: TemplateFormat = Field(..., alias="templateFormat", description="Template format")
    invocation_parameters: InvocationParameters = Field(..., alias="invocationParameters", description="Invocation parameters")
    tools: Optional[Tools] = Field(None, description="Tools")
    response_format: Optional[Dict[str, Any]] = Field(None, alias="responseFormat", description="Response format (provider-specific, pass-through)")
    created_at: Optional[str] = Field(None, alias="createdAt", description="Creation timestamp (ISO 8601)")
    updated_at: Optional[str] = Field(None, alias="updatedAt", description="Update timestamp (ISO 8601)")
    
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)
    
    @model_validator(mode='before')
    @classmethod
    def wrap_tools_array(cls, data):
        """Wrap tools array in Tools object. No normalization - pass-through user input."""
        if isinstance(data, dict) and "tools" in data:
            tools_value = data["tools"]
            if isinstance(tools_value, list):
                data["tools"] = {"type": "tools", "tools": tools_value}
            elif isinstance(tools_value, dict) and "tools" in tools_value:
                # Already wrapped
                pass
        return data


# ============================================================================
# LLM Service Request DTO
# ============================================================================

class LLMServiceRequestDto(BaseModel):
    model_configuration: ModelConfigurationWithEncryptedKey = Field(
        ..., description="Model configuration with decrypted API key (for LLM calls)"
    )
    prompt_version: PromptVersionDto = Field(..., description="Prompt version DTO")
    inputs: Dict[str, Any] = Field(..., description="Input variables for template rendering")
    
    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# LLM Service Response DTO
# ============================================================================

class TokenUsage(BaseModel):
    prompt_tokens: int = Field(..., alias="promptTokens", description="Number of prompt tokens")
    completion_tokens: int = Field(..., alias="completionTokens", description="Number of completion tokens")
    total_tokens: int = Field(..., alias="totalTokens", description="Total number of tokens")
    
    model_config = ConfigDict(populate_by_name=True)


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")


class ToolCall(BaseModel):
    id: str = Field(..., description="Tool call ID")
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")


class ExecutionMetadata(BaseModel):
    execution_time_ms: int = Field(..., alias="executionTimeMs", description="Execution time in milliseconds")
    finish_reason: Optional[str] = Field(None, alias="finishReason", description="Finish reason")
    
    model_config = ConfigDict(populate_by_name=True)


class LLMServiceResponseDto(BaseModel):
    output: str = Field(..., description="Generated output text")
    usage: Optional[TokenUsage] = Field(None, description="Token usage")
    model: Optional[ModelInfo] = Field(None, description="Model information")
    metadata: Optional[ExecutionMetadata] = Field(None, description="Execution metadata")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls")


# ============================================================================
# Experiment Worker Types
# ============================================================================

class ExperimentJobDto(BaseModel):
    """Job data format for experiment queue.
    Worker fetches latest prompt version via API, then model config from prompt version's modelConfigurationId.
    """
    experiment_id: str = Field(..., alias="experimentId", description="UUID of the experiment")
    dataset_row_id: str = Field(..., alias="datasetRowId", description=DATASET_ROW_ID_DESCRIPTION)
    prompt_id: str = Field(
        ...,
        alias="promptId",
        description="Prompt ID (UUID) - worker fetches latest version via API",
    )
    inputs: Dict[str, Any] = Field(
        ..., description="Input variables for template rendering"
    )
    message_id: Optional[str] = Field(None, alias="messageId", description="Message ID composed by Node app for correlation and logging")

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class ExperimentResultDto(BaseModel):
    """Result format for experiment-results-queue"""
    experiment_id: str = Field(..., alias="experimentId", description="UUID of the experiment")
    dataset_row_id: str = Field(..., alias="datasetRowId", description=DATASET_ROW_ID_DESCRIPTION)
    result: str = Field(..., description="The LLM output text")
    metadata: Optional[Dict[str, Union[int, str]]] = Field(None, description="Optional metadata (execution_time_ms, tokens_used, etc.)")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    message_id: Optional[str] = Field(None, alias="messageId", description="Message ID for correlation and logging (passed through from job)")
    
    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Evaluation Worker Types
# ============================================================================

class EvaluationJobDto(BaseModel):
    """Job data format for evaluation queue (RAGAS or LLM-based).
    For LLM-based: worker fetches latest prompt version via API, then model config from prompt version's modelConfigurationId.
    """
    evaluation_id: str = Field(..., alias="evaluationId", description="UUID of the evaluation")
    score_id: str = Field(..., alias="scoreId", description="UUID of the score")
    scoring_type: str = Field(..., alias="scoringType", description="RAGAS or LLM-based")
    dataset_row_id: str = Field(..., alias="datasetRowId", description=DATASET_ROW_ID_DESCRIPTION)
    experiment_result_id: Optional[str] = Field(None, alias="experimentResultId", description="UUID of the experiment result (null for dataset-scoped evaluations)")
    # RAGAS-specific
    ragas_model_configuration_id: Optional[str] = Field(None, alias="ragasModelConfigurationId")
    ragas_score_key: Optional[str] = Field(None, alias="ragasScoreKey")
    # LLM-based specific (worker fetches prompt version and model config via API)
    prompt_id: Optional[str] = Field(None, alias="promptId")
    # Common (inputs for evaluation)
    score_mapping: Dict[str, Any] = Field(default_factory=dict, alias="scoreMapping")
    message_id: Optional[str] = Field(None, alias="messageId", description="Message ID composed by Node app for correlation and logging")

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class EvaluationResultDto(BaseModel):
    """Result format for evaluation-results-queue."""
    evaluation_id: str = Field(..., alias="evaluationId", description="UUID of the evaluation")
    score_id: str = Field(..., alias="scoreId", description="UUID of the score")
    dataset_row_id: str = Field(..., alias="datasetRowId", description=DATASET_ROW_ID_DESCRIPTION)
    experiment_result_id: Optional[str] = Field(None, alias="experimentResultId", description="UUID of the experiment result (null for dataset-scoped evaluations)")
    score: Optional[str] = Field(None, description="Computed score (None on error)")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    metric: Optional[str] = Field(None, description="RAGAS metric name")
    metric_id: Optional[str] = Field(None, alias="metricId", description="RAGAS metric ID")
    message_id: Optional[str] = Field(None, alias="messageId", description="Message ID for correlation and logging (passed through from job)")

    model_config = ConfigDict(populate_by_name=True)
