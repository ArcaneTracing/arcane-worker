"""
Unit tests for registry pattern.
"""
import pytest
from app.core.registry import Registry, FactoryRegistry


class TestRegistry:
    """Tests for Registry class."""
    
    def test_register_and_get(self):
        """Should register and retrieve values."""
        registry = Registry()
        registry.register("key1", "value1")
        assert registry.get("key1") == "value1"
    
    def test_get_returns_none_for_missing_key(self):
        """Should return None for missing key."""
        registry = Registry()
        assert registry.get("missing") is None
    
    def test_raises_on_duplicate_register(self):
        """Should raise ValueError when registering duplicate key."""
        registry = Registry()
        registry.register("key1", "value1")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("key1", "value2")
    
    def test_has_checks_key_existence(self):
        """Should check if key exists."""
        registry = Registry()
        registry.register("key1", "value1")
        assert registry.has("key1") is True
        assert registry.has("missing") is False
    
    def test_get_all_keys(self):
        """Should return all registered keys."""
        registry = Registry()
        registry.register("key1", "value1")
        registry.register("key2", "value2")
        keys = registry.get_all_keys()
        assert set(keys) == {"key1", "key2"}
    
    def test_get_all_items(self):
        """Should return all registered items."""
        registry = Registry()
        registry.register("key1", "value1")
        registry.register("key2", "value2")
        items = registry.get_all_items()
        assert items == {"key1": "value1", "key2": "value2"}


class TestFactoryRegistry:
    """Tests for FactoryRegistry class."""
    
    def test_create_with_valid_key(self):
        """Should create instance using factory function."""
        registry = FactoryRegistry()
        
        def factory(x, y):
            return x + y
        
        registry.register("add", factory)
        result = registry.create("add", 2, 3)
        assert result == 5
    
    def test_create_returns_none_for_missing_key(self):
        """Should return None for missing key."""
        registry = FactoryRegistry()
        assert registry.create("missing") is None
    
    def test_create_or_raise_raises_for_missing_key(self):
        """Should raise ValueError for missing key."""
        registry = FactoryRegistry()
        with pytest.raises(ValueError, match="Unknown key"):
            registry.create_or_raise("missing")
    
    def test_create_or_raise_includes_available_keys(self):
        """Should include available keys in error message."""
        registry = FactoryRegistry()
        registry.register("key1", lambda: 1)
        registry.register("key2", lambda: 2)
        
        with pytest.raises(ValueError, match="key1.*key2"):
            registry.create_or_raise("missing")
    
    def test_create_with_keyword_arguments(self):
        """Should pass keyword arguments to factory."""
        registry = FactoryRegistry()
        
        def factory(x, y=10):
            return x + y
        
        registry.register("add", factory)
        result = registry.create("add", 5, y=3)
        assert result == 8

