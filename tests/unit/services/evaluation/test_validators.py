"""
Unit tests for RAGAS validators.
"""
import pytest
from app.services.evaluation.ragas.validators import (
    convert_to_string,
    convert_string_to_list,
    convert_list_to_string
)


class TestConvertToString:
    """Tests for convert_to_string function."""
    
    def test_converts_string(self):
        """Should return string as-is."""
        assert convert_to_string("test") == "test"
    
    def test_converts_integer(self):
        """Should convert integer to string."""
        assert convert_to_string(42) == "42"
    
    def test_converts_float(self):
        """Should convert float to string."""
        assert convert_to_string(3.14) == "3.14"
    
    def test_converts_none_to_empty_string(self):
        """Should convert None to empty string."""
        assert convert_to_string(None) == ""


class TestConvertStringToList:
    """Tests for convert_string_to_list function."""
    
    def test_converts_list_to_list(self):
        """Should return list as-is."""
        assert convert_string_to_list(["a", "b"]) == ["a", "b"]
    
    def test_converts_string_to_single_item_list(self):
        """Should convert string to single-item list."""
        assert convert_string_to_list("test") == ["test"]
    
    def test_handles_empty_string(self):
        """Should convert empty string to single-item list."""
        assert convert_string_to_list("") == [""]


class TestConvertListToString:
    """Tests for convert_list_to_string function."""
    
    def test_converts_list_to_first_element(self):
        """Should convert list to first element."""
        assert convert_list_to_string(["first", "second"]) == "first"
    
    def test_converts_empty_list_to_empty_string(self):
        """Should convert empty list to empty string."""
        assert convert_list_to_string([]) == ""
    
    def test_converts_string_to_string(self):
        """Should return string as-is."""
        assert convert_list_to_string("test") == "test"
    
    def test_converts_none_to_empty_string(self):
        """Should convert None to empty string."""
        assert convert_list_to_string(None) == ""

