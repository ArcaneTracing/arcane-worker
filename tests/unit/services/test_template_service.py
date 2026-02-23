"""
Unit tests for template service.
"""
import pytest
from app.services.template import TemplateService
from app.models.schemas import (
    TemplateFormat,
    PromptStringTemplate,
    PromptChatTemplate,
    PromptMessage
)


class TestTemplateService:
    """Tests for TemplateService."""
    
    def test_renders_string_template_mustache(self):
        """Should render Mustache string template."""
        service = TemplateService()
        template = PromptStringTemplate(template="Hello {{name}}!")
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.MUSTACHE,
            inputs=inputs
        )
        
        assert result.template == "Hello World!"
    
    def test_renders_string_template_f_string(self):
        """Should render f-string style template."""
        service = TemplateService()
        template = PromptStringTemplate(template="Hello {name}!")
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert result.template == "Hello World!"
    
    def test_renders_string_template_with_format_spec(self):
        """Should render f-string template with format specifier."""
        service = TemplateService()
        template = PromptStringTemplate(template="Value: {number:>5}")
        inputs = {"number": 42}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert result.template == "Value:    42"
    
    def test_renders_chat_template(self):
        """Should render chat template."""
        service = TemplateService()
        template = PromptChatTemplate(messages=[
            PromptMessage(role="user", content="Hello {name}!")
        ])
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert result.messages[0].content == "Hello World!"
    
    def test_returns_template_unchanged_for_none_format(self):
        """Should return template unchanged for NONE format."""
        service = TemplateService()
        template = PromptStringTemplate(template="Hello {{name}}!")
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.NONE,
            inputs=inputs
        )
        
        assert result.template == "Hello {{name}}!"
    
    def test_handles_missing_variable(self):
        """Should handle missing variable gracefully."""
        service = TemplateService()
        template = PromptStringTemplate(template="Hello {name}!")
        inputs = {}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        # Should use empty string for missing variable
        assert "Hello" in result.template
    
    
    def test_render_handles_format_errors(self):
        """Should handle formatting errors gracefully."""
        service = TemplateService()
        template = PromptStringTemplate(template="Value: {value:invalid}")
        inputs = {"value": 10}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        # Should fall back to string representation
        assert "10" in result.template

