"""
Unit tests for core validators.
"""
import pytest
from app.core.validators import (
    validate_required_field,
    validate_config_structure,
    extract_config_section,
    ValidationError
)


class TestValidateRequiredField:
    """Tests for validate_required_field function."""
    
    def test_validates_non_empty_string(self):
        """Should not raise for non-empty string."""
        validate_required_field("test", "field_name")
    
    def test_validates_non_empty_integer(self):
        """Should not raise for non-empty integer."""
        validate_required_field(42, "field_name")
    
    def test_raises_for_none(self):
        """Should raise ValidationError for None."""
        with pytest.raises(ValidationError, match="field_name is required"):
            validate_required_field(None, "field_name")
    
    def test_raises_for_empty_string(self):
        """Should raise ValidationError for empty string."""
        with pytest.raises(ValidationError, match="field_name is required"):
            validate_required_field("", "field_name")
    
    def test_includes_context_in_error(self):
        """Should include context in error message."""
        with pytest.raises(ValidationError, match="in context"):
            validate_required_field(None, "field_name", context="context")


class TestValidateConfigStructure:
    """Tests for validate_config_structure function."""
    
    def test_validates_complete_config(self):
        """Should not raise for config with all required keys."""
        config = {"key1": "value1", "key2": "value2", "key3": "value3"}
        validate_config_structure(config, ["key1", "key2"])
    
    def test_raises_for_missing_key(self):
        """Should raise ValidationError for missing key."""
        config = {"key1": "value1"}
        with pytest.raises(ValidationError, match="missing required keys"):
            validate_config_structure(config, ["key1", "key2"])
    
    def test_raises_for_none_value(self):
        """Should raise ValidationError for None value."""
        config = {"key1": "value1", "key2": None}
        with pytest.raises(ValidationError, match="missing required keys"):
            validate_config_structure(config, ["key1", "key2"])
    
    def test_includes_config_name_in_error(self):
        """Should include config name in error message."""
        config = {}
        with pytest.raises(ValidationError, match="test_config"):
            validate_config_structure(config, ["key1"], config_name="test_config")


class TestExtractConfigSection:
    """Tests for extract_config_section function."""
    
    def test_extracts_existing_section(self):
        """Should extract existing section."""
        config = {"section": {"key": "value"}}
        result = extract_config_section(config, "section")
        assert result == {"key": "value"}
    
    def test_raises_for_missing_section(self):
        """Should raise ValidationError for missing section."""
        config = {"other_section": {}}
        with pytest.raises(ValidationError, match="missing 'section' key"):
            extract_config_section(config, "section")
    
    def test_raises_for_non_dict_section(self):
        """Should raise ValidationError for non-dict section."""
        config = {"section": "not_a_dict"}
        with pytest.raises(ValidationError, match="must be a dictionary"):
            extract_config_section(config, "section")
    
    def test_includes_section_name_in_error(self):
        """Should include section name in error message."""
        config = {}
        with pytest.raises(ValidationError, match="test_section"):
            extract_config_section(config, "section", section_name="test_section")

