"""
Patch instructor.from_genai to use use_async=True for RAGAS compatibility.

RAGAS metrics call llm.agenerate() which requires an async client. The instructor
library wraps google-genai Client with use_async=False by default, producing a
sync Instructor. This patch forces use_async=True so RAGAS receives AsyncInstructor.

Must be imported before any code that builds RAGAS LLMs with Google clients.
"""
import logging

logger = logging.getLogger(__name__)


def apply_instructor_genai_async_patch() -> None:
    """Patch instructor.from_genai to use use_async=True for async RAGAS support."""
    try:
        import instructor
        from instructor.providers.genai import client as genai_client

        _original_from_genai = genai_client.from_genai

        def _patched_from_genai(client, mode=None, use_async=False, **kwargs):
            # Force async for RAGAS compatibility (agenerate requires async client)
            if mode is None:
                mode = instructor.Mode.GENAI_TOOLS
            return _original_from_genai(client, mode=mode, use_async=True, **kwargs)

        genai_client.from_genai = _patched_from_genai
        instructor.from_genai = _patched_from_genai
        logger.debug("Applied instructor.from_genai async patch for RAGAS")
    except ImportError as e:
        logger.debug("Skipping instructor genai patch (instructor or genai not installed): %s", e)
