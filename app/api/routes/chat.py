"""
Chat API routes using FastAPI.
"""
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Depends

from app.api.dependencies import get_model_service
from app.models.schemas import (
    LLMServiceRequestDto,
    LLMServiceResponseDto,
    ExecutionMetadata,
    TokenUsage,
    ModelInfo,
    ToolCall,
)

router = APIRouter()


def _build_usage(usage_dict: Any) -> TokenUsage | None:
    """Convert usage dict to TokenUsage model."""
    if not usage_dict:
        return None
    return TokenUsage(**usage_dict) if isinstance(usage_dict, dict) else usage_dict


def _build_model_info(model_dict: Any) -> ModelInfo | None:
    """Convert model dict to ModelInfo model."""
    if not model_dict:
        return None
    return ModelInfo(**model_dict) if isinstance(model_dict, dict) else model_dict


def _build_tool_calls(tool_calls_raw: Any) -> list[ToolCall] | None:
    """Convert tool_calls list to ToolCall models."""
    if not tool_calls_raw:
        return None
    return [
        ToolCall(**tc) if isinstance(tc, dict) else tc
        for tc in tool_calls_raw
    ]


def _response_to_dto(response: dict[str, Any], execution_time_ms: int) -> LLMServiceResponseDto:
    """Convert raw response dict to LLMServiceResponseDto."""
    usage = _build_usage(response.get("usage"))
    model_info = _build_model_info(response.get("model"))
    metadata = ExecutionMetadata(
        execution_time_ms=execution_time_ms,
        finish_reason=response.get("finish_reason"),
    )
    tool_calls = _build_tool_calls(response.get("tool_calls"))
    return LLMServiceResponseDto(
        output=response.get("output", ""),
        usage=usage,
        model=model_info,
        metadata=metadata,
        tool_calls=tool_calls,
    )


def _extract_error_detail(exc: Exception) -> tuple[int, str]:
    """Extract status code and detail message from exception."""
    status_code = getattr(exc, "status_code", None) or 500
    detail = str(exc)
    body = getattr(exc, "body", None)
    if body is None and hasattr(exc, "response") and exc.response is not None:
        try:
            body = getattr(exc.response, "json", lambda: {})() or {}
        except Exception:
            body = {}
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message", None) if isinstance(err, dict) else None
        if msg:
            detail = msg
    return status_code, detail


@router.post("/run", response_model=LLMServiceResponseDto)
async def run(
    request: LLMServiceRequestDto,
    model_service=Depends(get_model_service),
) -> LLMServiceResponseDto:
    """
    Execute an LLM service request with template rendering.

    The model service handles:
    - Template rendering (Mustache or F-string format)
    - Message conversion (provider-specific format)
    - Parameter extraction (temperature, max_tokens, etc.)
    - LLM execution (OpenAI, Anthropic, Azure, etc.)

    Args:
        request: LLM service request containing model configuration,
                 prompt version, and input variables
        model_service: Model service instance (injected via FastAPI dependency)

    Returns:
        LLMServiceResponseDto containing:
        - output: Generated text response
        - usage: Token usage statistics (if available)
        - model: Model information (if available)
        - metadata: Execution metadata (execution time, finish reason)
        - tool_calls: Tool calls if the model requested function calls

    Raises:
        HTTPException:
            - 400: Bad request (invalid input or validation error)
            - 500: Internal server error (LLM execution failed)
    """
    start_time = time.time()
    try:
        response = await model_service.execute(request)
        execution_time_ms = int((time.time() - start_time) * 1000)
        return _response_to_dto(response, execution_time_ms)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        from app.core.error_handling import log_api_error

        log_api_error(error=e, service_name="ChatAPI", context="run()")
        status_code, detail = _extract_error_detail(e)
        raise HTTPException(status_code=status_code, detail=detail) from e

