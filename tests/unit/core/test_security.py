"""
Unit tests for security utilities.
"""
import pytest
from app.core.security import (
    validate_json_size,
    safe_json_loads,
    safe_from_json,
    validate_string_length,
    validate_dict_size,
    validate_list_length,
    sanitize_format_spec,
    SecurityError,
    MAX_JSON_SIZE,
    MAX_STRING_LENGTH,
    MAX_DICT_KEYS,
    MAX_LIST_LENGTH
)


class TestValidateJsonSize:
    """Tests for validate_json_size function."""
    
    def test_validates_small_json(self):
        """Should not raise for JSON within size limit."""
        small_json = '{"key": "value"}'
        validate_json_size(small_json)
    
    def test_raises_for_large_json(self):
        """Should raise SecurityError for JSON exceeding size limit."""
        large_json = "x" * (MAX_JSON_SIZE + 1)
        with pytest.raises(SecurityError, match="exceeds maximum allowed size"):
            validate_json_size(large_json)
    
    def test_uses_custom_max_size(self):
        """Should use custom max_size if provided."""
        json_str = "x" * 100
        validate_json_size(json_str, max_size=200)  # Should pass
        with pytest.raises(SecurityError):
            validate_json_size(json_str, max_size=50)  # Should fail


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""
    
    def test_parses_valid_json(self):
        """Should parse valid JSON."""
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_returns_fallback_for_invalid_json(self):
        """Should return fallback for invalid JSON."""
        result = safe_json_loads("invalid json", fallback={"error": True})
        assert result == {"error": True}
    
    def test_returns_fallback_for_large_json(self):
        """Should return fallback for JSON exceeding size limit."""
        large_json = "x" * (MAX_JSON_SIZE + 1)
        result = safe_json_loads(large_json, fallback=None)
        assert result is None
    
    def test_handles_non_string_input(self):
        """Should return non-string input as-is."""
        result = safe_json_loads({"already": "parsed"})
        assert result == {"already": "parsed"}


class TestSafeFromJson:
    """Tests for safe_from_json function."""
    
    def test_parses_valid_json(self):
        """Should parse valid JSON using Pydantic's from_json."""
        result = safe_from_json('{"key": "value"}', fallback=None)
        assert result == {"key": "value"}
    
    def test_returns_fallback_for_invalid_json(self):
        """Should return fallback for invalid JSON."""
        result = safe_from_json("invalid json", fallback={"error": True})
        assert result == {"error": True}
    
    def test_handles_partial_json(self):
        """Should handle partial JSON with allow_partial=True."""
        partial_json = '{"key": "value"'  # Missing closing brace
        result = safe_from_json(partial_json, allow_partial=True, fallback=None)
        # Should attempt to parse and return fallback if parsing fails
        assert result is not None or result == {"error": True}


class TestValidateStringLength:
    """Tests for validate_string_length function."""
    
    def test_validates_short_string(self):
        """Should not raise for string within length limit."""
        validate_string_length("short string")
    
    def test_raises_for_long_string(self):
        """Should raise SecurityError for string exceeding length limit."""
        long_string = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(SecurityError, match="exceeds maximum allowed length"):
            validate_string_length(long_string)
    
    def test_uses_custom_max_length(self):
        """Should use custom max_length if provided."""
        string = "x" * 100
        validate_string_length(string, max_length=200)  # Should pass
        with pytest.raises(SecurityError):
            validate_string_length(string, max_length=50)  # Should fail


class TestValidateDictSize:
    """Tests for validate_dict_size function."""
    
    def test_validates_small_dict(self):
        """Should not raise for dict within size limit."""
        small_dict = {f"key{i}": f"value{i}" for i in range(10)}
        validate_dict_size(small_dict)
    
    def test_raises_for_large_dict(self):
        """Should raise SecurityError for dict exceeding size limit."""
        large_dict = {f"key{i}": f"value{i}" for i in range(MAX_DICT_KEYS + 1)}
        with pytest.raises(SecurityError, match="exceeds maximum allowed size"):
            validate_dict_size(large_dict)


class TestValidateListLength:
    """Tests for validate_list_length function."""
    
    def test_validates_short_list(self):
        """Should not raise for list within length limit."""
        validate_list_length([1, 2, 3])
    
    def test_raises_for_long_list(self):
        """Should raise SecurityError for list exceeding length limit."""
        long_list = list(range(MAX_LIST_LENGTH + 1))
        with pytest.raises(SecurityError, match="exceeds maximum allowed length"):
            validate_list_length(long_list)
    
    def test_uses_custom_max_length_for_list(self):
        """Should use custom max_length if provided."""
        test_list = list(range(100))
        validate_list_length(test_list, max_length=200)  # Should pass
        with pytest.raises(SecurityError):
            validate_list_length(test_list, max_length=50)  # Should fail


class TestSanitizeFormatSpec:
    """Tests for sanitize_format_spec function."""
    
    def test_allows_safe_format_specs(self):
        """Should allow safe format specifiers."""
        assert sanitize_format_spec(".2f") == ".2f"
        assert sanitize_format_spec(">10") == ">10"
        assert sanitize_format_spec("0.3f") == "0.3f"
    
    def test_raises_for_unsafe_format_specs(self):
        """Should raise SecurityError for unsafe format specifiers."""
        unsafe_specs = ["!r", ".upper()", "(format)", "[0]", "{key}", "?@", "|&"]
        for spec in unsafe_specs:
            with pytest.raises(SecurityError, match="Unsafe format specifier"):
                sanitize_format_spec(spec)

