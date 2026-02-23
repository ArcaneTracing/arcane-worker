"""
Template rendering service for prompt templates.

Supports multiple template formats:
- Mustache templates ({{variable}})
- F-string templates ({variable} or {variable:format})
- NONE (no rendering)

Handles both string templates and chat templates with message lists.
"""
from __future__ import annotations

from typing import Dict, List, Match, Any
import logging
from app.models.schemas import (
    PromptTemplate,
    PromptStringTemplate,
    PromptChatTemplate,
    TemplateFormat,
    PromptMessage,
    ContentPart,
)
import re
from pystache import Renderer

logger = logging.getLogger(__name__)


class TemplateService:
    """Service for rendering prompt templates with different formats"""
    
    def __init__(self) -> None:
        self.mustache_renderer: Renderer = Renderer()
    
    def render_template(
        self,
        template: PromptTemplate,
        template_format: TemplateFormat,
        inputs: Dict[str, Any]
    ) -> PromptTemplate:
        """
        Render a template using the specified format and inputs.
        
        Args:
            template: The template to render
            template_format: The format of the template (MUSTACHE, F_STRING, NONE)
            inputs: Input variables for template rendering
            
        Returns:
            Rendered template
        """
        if template_format == TemplateFormat.NONE:
            return template
        
        if isinstance(template, PromptStringTemplate):
            rendered_content = self._render_string(
                template.template,
                template_format,
                inputs
            )
            return PromptStringTemplate(template=rendered_content)
        
        elif isinstance(template, PromptChatTemplate):
            rendered_messages = []
            for message in template.messages:
                rendered_message = self._render_message(message, template_format, inputs)
                rendered_messages.append(rendered_message)
            return PromptChatTemplate(messages=rendered_messages)
        
        return template
    
    def _render_string(
        self,
        content: str,
        template_format: TemplateFormat,
        inputs: Dict[str, Any]
    ) -> str:
        """Render a string template"""
        if template_format == TemplateFormat.MUSTACHE:
            return self.mustache_renderer.render(content, inputs)
        elif template_format == TemplateFormat.F_STRING:
            # Use f-string style formatting with safe evaluation
            # Replace {variable} with values from inputs
            return self._render_f_string(content, inputs)
        else:
            return content
    
    def _render_f_string(self, template: str, inputs: Dict[str, Any]) -> str:
        """
        Render f-string style template safely.
        Handles {variable} and {variable:format} patterns.
        
        Security: Validates format specifiers to prevent code injection.
        """
        from app.core.security import sanitize_format_spec, SecurityError
        
        def replace_var(match: Match[str]) -> str:
            var_expr = match.group(1)
            # Handle format specifiers like {var:format}
            if ':' in var_expr:
                var_name, format_spec = var_expr.split(':', 1)
                value = inputs.get(var_name, '')
                try:
                    # Sanitize format specifier to prevent code injection
                    safe_format_spec = sanitize_format_spec(format_spec)
                    return f"{value:{safe_format_spec}}"
                except SecurityError:
                    # If format spec is unsafe, fall back to string representation
                    logger.warning(f"Unsafe format specifier in template, using string representation: {format_spec}")
                    return str(value)
                except (ValueError, TypeError) as e:
                    # Handle formatting errors gracefully
                    logger.debug(f"Formatting error for value {value} with spec {format_spec}: {e}")
                    return str(value)
            else:
                return str(inputs.get(var_expr, ''))
        
        # Match {variable} or {variable:format}
        pattern = r'\{([^}]+)\}'
        return re.sub(pattern, replace_var, template)
    
    def _render_content_part(
        self,
        part: ContentPart,
        template_format: TemplateFormat,
        inputs: Dict[str, Any]
    ) -> ContentPart:
        """Render a single ContentPart, handling both dict and Pydantic model formats."""
        # Extract text from either dict or Pydantic model
        text = None
        if isinstance(part, dict):
            text = part.get('text')
        elif hasattr(part, 'text'):
            text = part.text
        
        # If no text found, return part as-is
        if text is None:
            return part
        
        # Render the text and return updated part
        rendered_text = self._render_string(text, template_format, inputs)
        return {'text': rendered_text}
    
    def _render_message(
        self,
        message: PromptMessage,
        template_format: TemplateFormat,
        inputs: Dict[str, Any]
    ) -> PromptMessage:
        """Render a message template"""
        if isinstance(message.content, str):
            rendered_content = self._render_string(message.content, template_format, inputs)
            return PromptMessage(role=message.role, content=rendered_content)
        
        if isinstance(message.content, list):
            # Handle ContentPart list
            rendered_parts = [
                self._render_content_part(part, template_format, inputs)
                for part in message.content
            ]
            return PromptMessage(role=message.role, content=rendered_parts)
        
        return message

