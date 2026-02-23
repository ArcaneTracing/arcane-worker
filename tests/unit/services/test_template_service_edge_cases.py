"""
Additional unit tests for template service edge cases.
"""
import pytest
from app.services.template import TemplateService
from app.models.schemas import (
    TemplateFormat,
    PromptStringTemplate,
    PromptChatTemplate,
    PromptMessage,
    TextContentPart
)


class TestTemplateServiceEdgeCases:
    """Edge case tests for TemplateService to improve coverage."""
    
    @pytest.fixture
    def service(self):
        """Template service fixture."""
        return TemplateService()
    
    def test_render_message_with_dict_content_parts_no_text(self, service):
        """Should preserve non-text content parts."""
        template = PromptChatTemplate(messages=[
            PromptMessage(
                role="user",
                content=[{"tool_call": {"name": "test"}}]  # ContentPart without text
            )
        ])
        inputs = {}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        # Should preserve non-text content parts
        rendered_content = result.messages[0].content
        assert isinstance(rendered_content, list)
        assert len(rendered_content) == 1
    
    def test_render_f_string_with_format_specifier(self, service):
        """Should handle format specifiers in f-string templates."""
        template = PromptStringTemplate(template="Value: {number:.2f}")
        inputs = {"number": 3.14159}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert "3.14" in result.template
    
    def test_render_f_string_with_unsafe_format_specifier(self, service):
        """Should handle unsafe format specifiers gracefully."""
        template = PromptStringTemplate(template="Value: {value:__import__('os')}")
        inputs = {"value": "test"}
        
        # Should not raise error, but fall back to string representation
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert result is not None
        assert isinstance(result, PromptStringTemplate)
    
    def test_render_message_with_dict_content_text(self, service):
        """Should render text in dict content parts."""
        template = PromptChatTemplate(messages=[
            PromptMessage(
                role="user",
                content=[{"text": "Hello {name}!"}]
            )
        ])
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        rendered_content = result.messages[0].content
        assert isinstance(rendered_content, list)
        assert len(rendered_content) == 1
        # Content may be converted to Pydantic model or dict
        if hasattr(rendered_content[0], 'text'):
            assert rendered_content[0].text == "Hello World!"
        else:
            assert rendered_content[0]["text"] == "Hello World!"
    
    def test_render_message_with_pydantic_content_part_no_text(self, service):
        """Should preserve non-text content parts."""
        # Test that non-text content parts are preserved
        # Use ToolCallContentPart representation without text
        template = PromptChatTemplate(messages=[
            PromptMessage(
                role="user",
                content=[{"tool_call": {"name": "func", "arguments": {}}}]
            )
        ])
        inputs = {}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        rendered_content = result.messages[0].content
        assert isinstance(rendered_content, list)
        # Should preserve the part even if it doesn't have text
        assert len(rendered_content) == 1
    
    def test_render_string_mustache(self, service):
        """Should render Mustache template."""
        template = PromptStringTemplate(template="Hello {{name}}!")
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.MUSTACHE,
            inputs=inputs
        )
        
        assert result.template == "Hello World!"
    
    def test_render_string_f_string(self, service):
        """Should render f-string template."""
        template = PromptStringTemplate(template="Hello {name}!")
        inputs = {"name": "World"}
        
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.F_STRING,
            inputs=inputs
        )
        
        assert result.template == "Hello World!"
    
    def test_render_string_other_format(self, service):
        """Should return content unchanged for unknown format."""
        template = PromptStringTemplate(template="Hello {name}!")
        inputs = {"name": "World"}
        
        # TemplateFormat.NONE should return unchanged
        result = service.render_template(
            template=template,
            template_format=TemplateFormat.NONE,
            inputs=inputs
        )
        
        assert result.template == "Hello {name}!"

