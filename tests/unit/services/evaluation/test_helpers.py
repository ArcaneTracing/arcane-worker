"""
Unit tests for RAGAS helpers.
"""
import pytest
from app.services.evaluation.ragas.helpers import extract_score
from unittest.mock import Mock


class TestExtractScore:
    """Tests for extract_score function."""
    
    def test_extracts_float(self):
        """Should return float as-is."""
        assert extract_score(0.85) == 0.85
    
    def test_extracts_integer(self):
        """Should return integer as-is."""
        assert extract_score(42) == 42
    
    def test_extracts_from_object_value(self):
        """Should extract from object with .value attribute."""
        result = Mock()
        result.value = 0.75
        
        assert extract_score(result) == 0.75
    
    def test_handles_zero_score(self):
        """Should handle zero score."""
        assert extract_score(0) == 0
    
    def test_handles_negative_score(self):
        """Should handle negative score."""
        assert extract_score(-1) == -1

