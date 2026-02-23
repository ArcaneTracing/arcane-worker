"""
Unit tests for JSON parser.
"""
import pytest
from app.services.evaluation.ragas.json_parser import parse_json_strings


class TestParseJsonStrings:
    """Tests for parse_json_strings function."""
    
    def test_parses_json_string_in_dict(self):
        """Should parse JSON string in dictionary."""
        data = {"key": '{"nested": "value"}'}
        result = parse_json_strings(data)
        
        assert isinstance(result["key"], dict)
        assert result["key"]["nested"] == "value"
    
    def test_parses_json_string_in_list(self):
        """Should parse JSON string in list."""
        data = ['{"key": "value"}', "normal_string"]
        result = parse_json_strings(data)
        
        assert isinstance(result[0], dict)
        assert result[0]["key"] == "value"
        assert result[1] == "normal_string"
    
    def test_handles_nested_json_strings(self):
        """Should handle nested JSON strings."""
        data = {"outer": '{"inner": {"deep": "value"}}'}  # Valid nested JSON
        result = parse_json_strings(data)
        
        # Should parse nested JSON structures
        assert isinstance(result["outer"], dict)
        assert isinstance(result["outer"].get("inner"), dict)
    
    def test_handles_non_json_strings(self):
        """Should leave non-JSON strings unchanged."""
        data = {"key": "normal string"}
        result = parse_json_strings(data)
        
        assert result["key"] == "normal string"
    
    def test_handles_partial_json(self):
        """Should handle partial/incomplete JSON."""
        data = {"key": '["incomplete list'}
        result = parse_json_strings(data)
        
        # Should attempt to extract content from partial JSON
        assert result is not None
    
    def test_handles_empty_string(self):
        """Should handle empty string."""
        data = {"key": ""}
        result = parse_json_strings(data)
        
        assert result["key"] == ""
    
    def test_handles_none(self):
        """Should handle None values."""
        data = {"key": None}
        result = parse_json_strings(data)
        
        assert result["key"] is None
    
    def test_handles_already_parsed_data(self):
        """Should handle already parsed data."""
        data = {"key": {"already": "parsed"}}
        result = parse_json_strings(data)
        
        assert result == data

