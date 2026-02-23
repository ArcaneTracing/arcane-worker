"""
Unit tests for message utilities.
"""
import pytest
from app.core.message_utils import (
    extract_text_from_content,
    normalize_role,
    convert_tools_to_format
)


class TestExtractTextFromContent:
    """Tests for extract_text_from_content function."""
    
    def test_extracts_string(self):
        """Should return string as-is."""
        result = extract_text_from_content("test string")
        assert result == "test string"
    
    def test_extracts_from_dict_list(self):
        """Should extract text from list of dict ContentParts."""
        content = [{"text": "part1"}, {"text": "part2"}]
        result = extract_text_from_content(content)
        assert result == "part1 part2"
    
    def test_extracts_from_object_list(self):
        """Should extract text from list of object ContentParts."""
        class ContentPart:
            def __init__(self, text):
                self.text = text
        
        content = [ContentPart("part1"), ContentPart("part2")]
        result = extract_text_from_content(content)
        assert result == "part1 part2"
    
    def test_handles_empty_list(self):
        """Should return empty string for empty list."""
        result = extract_text_from_content([])
        assert result == ""
    
    def test_handles_non_list_non_string(self):
        """Should return empty string for non-list non-string."""
        result = extract_text_from_content(None)
        assert result == ""
    
    def test_handles_mixed_content(self):
        """Should extract text from mixed content types."""
        class ContentPart:
            def __init__(self, text):
                self.text = text
        
        content = [{"text": "dict"}, ContentPart("object")]
        result = extract_text_from_content(content)
        assert "dict" in result and "object" in result


class TestNormalizeRole:
    """Tests for normalize_role function."""
    
    def test_keeps_supported_roles(self):
        """Should keep supported roles as-is."""
        assert normalize_role("system") == "system"
        assert normalize_role("user") == "user"
        assert normalize_role("assistant") == "assistant"
    
    def test_maps_unknown_to_user(self):
        """Should map unknown roles to user."""
        assert normalize_role("unknown") == "user"
        assert normalize_role("custom_role") == "user"
    
    def test_keeps_assistant(self):
        """Should keep assistant role."""
        assert normalize_role("assistant") == "assistant"
    
    def test_uses_custom_supported_roles(self):
        """Should use custom supported roles if provided."""
        supported = {"admin", "member"}
        assert normalize_role("admin", supported_roles=supported) == "admin"
        assert normalize_role("user", supported_roles=supported) == "user"  # Not in supported


class TestConvertToolsToFormat:
    """Tests for convert_tools_to_format function."""
    
    def test_returns_none_for_no_tools(self):
        """Should return None when no tools provided."""
        result = convert_tools_to_format(None)
        assert result is None
    
    def test_returns_none_for_empty_tools(self):
        """Should return None when tools.tools is empty."""
        class Tools:
            tools = []
        
        result = convert_tools_to_format(Tools())
        assert result is None
    
    def test_passes_through_tools(self):
        """Should pass-through tools without conversion."""
        class Tools:
            def __init__(self):
                self.tools = [
                    {"type": "function", "function": {"name": "func1"}},
                    {"type": "custom", "name": "func2", "input_schema": {}}
                ]
        
        result = convert_tools_to_format(Tools(), format_type="openai")
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[1]["type"] == "custom"

